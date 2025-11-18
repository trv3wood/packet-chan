"""Streamlit app: upload files, ingest into Chroma, ask queries."""
import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import time
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ingest import save_uploaded_file, extract_text_from_file, chunk_text
from vector_store import VectorStore
from retriever import build_prompt
from llm_adapter import generate, get_llm_status

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CONVERSATION_HISTORY_FILE = os.getenv("CONVERSATION_HISTORY_FILE", "./conversation_history.json")
CONVERSATION_THREADS_FILE = os.getenv("CONVERSATION_THREADS_FILE", "./conversation_threads.json")
MAX_CONVERSATION_THREADS = 5


@st.cache_resource
def get_store():
    return VectorStore(persist_directory=CHROMA_DIR)


def save_conversation_history(history):
    """保存对话历史到本地文件
    
    Args:
        history: 对话历史列表
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(CONVERSATION_HISTORY_FILE), exist_ok=True)
        
        # 保存对话历史
        with open(CONVERSATION_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.info(f"对话历史已保存到 {CONVERSATION_HISTORY_FILE}")
        return True
    except Exception as e:
        logger.error(f"保存对话历史失败: {e}")
        return False


def load_conversation_history():
    """从本地文件加载对话历史
    
    Returns:
        对话历史列表，如果文件不存在或加载失败则返回包含欢迎消息的默认列表
    """
    try:
        if os.path.exists(CONVERSATION_HISTORY_FILE):
            with open(CONVERSATION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
                logger.info(f"从 {CONVERSATION_HISTORY_FILE} 加载了 {len(history)} 条对话历史")
                return history
        else:
            logger.info(f"对话历史文件不存在: {CONVERSATION_HISTORY_FILE}")
    except Exception as e:
        logger.error(f"加载对话历史失败: {e}")
    
    # 返回默认对话历史
    return [
        {
            "role": "assistant",
            "content": "您好！我是您的AI助手。请上传文件并提问，我将基于文件内容为您提供答案。",
            "timestamp": "欢迎消息"
        }
    ]


def add_conversation_to_vector_store(history, store=None):
    """将对话历史添加到向量库中
    
    Args:
        history: 对话历史列表
        store: VectorStore实例，如果为None则自动获取
    
    Returns:
        添加到向量库的文档数量
    """
    if store is None:
        store = get_store()
    
    # 准备对话历史文档
    conversation_docs = []
    
    # 遍历对话历史，将问答对组合成文档
    i = 0
    while i < len(history):
        # 跳过欢迎消息和单独的助手/用户消息
        if history[i].get("role") == "user" and i + 1 < len(history) and history[i + 1].get("role") == "assistant":
            user_message = history[i]
            assistant_message = history[i + 1]
            
            # 组合问答对为一个文档
            conversation_text = f"用户问题: {user_message.get('content', '')}\n\nAI回答: {assistant_message.get('content', '')}"
            
            # 创建文档
            doc = {
                "id": f"conversation_{uuid.uuid4()}",
                "text": conversation_text,
                "metadata": {
                    "source": "conversation_history",
                    "timestamp": user_message.get("timestamp", ""),
                    "type": "conversation_pair"
                }
            }
            conversation_docs.append(doc)
            
            # 跳过已处理的助手消息
            i += 2
        else:
            i += 1
    
    # 如果有对话文档，添加到向量库
    if conversation_docs:
        try:
            logger.info(f"准备将 {len(conversation_docs)} 条对话历史添加到向量库")
            store.add_documents(conversation_docs)
            logger.info(f"成功将 {len(conversation_docs)} 条对话历史添加到向量库")
            return len(conversation_docs)
        except Exception as e:
            logger.error(f"将对话历史添加到向量库失败: {e}")
            return 0
    
    logger.info("没有找到可添加到向量库的对话历史")
    return 0


def save_conversation_threads(threads):
    """保存对话线程到本地文件
    
    Args:
        threads: 对话线程字典
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(CONVERSATION_THREADS_FILE), exist_ok=True)
        
        # 保存对话线程
        with open(CONVERSATION_THREADS_FILE, 'w', encoding='utf-8') as f:
            json.dump(threads, f, ensure_ascii=False, indent=2)
        logger.info(f"对话线程已保存到 {CONVERSATION_THREADS_FILE}")
        return True
    except Exception as e:
        logger.error(f"保存对话线程失败: {e}")
        return False


def load_conversation_threads():
    """从本地文件加载对话线程
    
    Returns:
        对话线程字典，如果文件不存在或加载失败则返回空字典
    """
    try:
        if os.path.exists(CONVERSATION_THREADS_FILE):
            with open(CONVERSATION_THREADS_FILE, 'r', encoding='utf-8') as f:
                threads = json.load(f)
                logger.info(f"从 {CONVERSATION_THREADS_FILE} 加载了 {len(threads)} 个对话线程")
                return threads
        else:
            logger.info(f"对话线程文件不存在: {CONVERSATION_THREADS_FILE}")
    except Exception as e:
        logger.error(f"加载对话线程失败: {e}")
    
    # 返回空字典
    return {}


def delete_conversation_thread(thread_id):
    """删除指定的对话线程并同步更新本地存储"""
    # 从会话状态中删除线程
    if thread_id in st.session_state.conversation_threads:
        del st.session_state.conversation_threads[thread_id]
        # 同步更新本地存储
        save_conversation_threads(st.session_state.conversation_threads)
        st.info(f"删除对话线程: {thread_id}")
        return True
    else:
        st.error(f"找不到对话线程: {thread_id}")
        return False


def create_new_conversation_thread(threads, thread_id=None):
    """创建新的对话线程
    
    Args:
        threads: 现有对话线程字典
        thread_id: 可选的线程ID，如果不提供则生成新ID
    
    Returns:
        (thread_id, new_thread): 线程ID和新线程对象
    """
    # 检查是否达到最大线程数
    if len(threads) >= MAX_CONVERSATION_THREADS:
        logger.warning(f"已达到最大对话线程数: {MAX_CONVERSATION_THREADS}")
        # 删除最旧的线程
        oldest_thread_id = min(threads.keys(), key=lambda k: threads[k].get('created_at', ''))
        del threads[oldest_thread_id]
        logger.info(f"删除了最旧的对话线程: {oldest_thread_id}")
    
    # 生成新的线程ID
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    # 创建新线程
    new_thread = {
        'id': thread_id,
        'name': '新对话',
        'created_at': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'last_updated': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'conversation_history': [
            {
                "role": "assistant",
                "content": "您好！我是您的AI助手。请上传文件并提问，我将基于文件内容为您提供答案。",
                "timestamp": "欢迎消息"
            }
        ]
    }
    
    # 添加到线程字典
    threads[thread_id] = new_thread
    logger.info(f"创建了新的对话线程: {thread_id}")
    
    # 保存到本地文件
    save_conversation_threads(threads)
    
    return thread_id, new_thread


def update_thread_name(threads, thread_id, user_question):
    """根据用户的首个提问更新线程名称
    
    Args:
        threads: 对话线程字典
        thread_id: 线程ID
        user_question: 用户的首个问题
    """
    if thread_id in threads:
        # 截取问题前20个字符作为线程名称
        thread_name = user_question[:20]
        if len(user_question) > 20:
            thread_name += '...'
        
        threads[thread_id]['name'] = thread_name
        threads[thread_id]['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 保存到本地文件
        save_conversation_threads(threads)
        logger.info(f"更新了对话线程名称: {thread_id} -> {thread_name}")
        return True
    return False


def ingest_and_index(uploaded_files):
    """处理多个文件并索引到向量数据库
    
    Args:
        uploaded_files: 单个文件或文件列表
    """
    # 确保是列表格式
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    
    # 限制最多5个文件
    if len(uploaded_files) > 5:
        st.error(f"最多只能处理5个文件，您上传了 {len(uploaded_files)} 个文件")
        uploaded_files = uploaded_files[:5]
    
    # 清除上一次上传的文件内容缓存
    store = get_store()
    
    # 清除 Streamlit 缓存
    get_store.clear()
    
    # 删除整个集合并重新创建（更彻底）
    store.delete_collection()
    
    # 重新获取 VectorStore 实例
    store = get_store()
    
    tmpdir = os.path.join(".", "uploads")
    total_docs = 0
    total_time = time.time()
    
    # 显示进度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_start_time = time.time()
        
        # 更新进度
        progress = (idx + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"正在处理文件 {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            # 保存文件
            path = save_uploaded_file(uploaded_file, tmpdir)
            
            # 提取文本
            extract_start = time.time()
            try:
                text = extract_text_from_file(path)
                # 检查提取的文本是否为空或太短
                if not text or len(text.strip()) < 10:
                    st.warning(f"⚠️ 文件 {uploaded_file.name} 提取的文本为空或太短（{len(text)} 字符），可能无法正确索引")
            except Exception as e:
                st.error(f"❌ 文件 {uploaded_file.name} 文本提取失败: {e}")
                continue
            extract_time = time.time() - extract_start
            
            # 分块
            chunk_start = time.time()
            chunks = chunk_text(text)
            chunk_time = time.time() - chunk_start
            
            if not chunks:
                st.warning(f"⚠️ 文件 {uploaded_file.name} 没有生成任何文档块")
                continue
            
            # 构建文档
            docs = []
            for i, c in enumerate(chunks):
                docs.append({
                    "id": f"{uuid.uuid4()}",
                    "text": c,
                    "metadata": {
                        "source": uploaded_file.name,
                        "chunk_index": i,
                        "file_index": idx
                    },
                })
            
            # 添加到向量数据库
            embed_start = time.time()
            store.add_documents(docs)
            embed_time = time.time() - embed_start
            
            total_docs += len(docs)
            file_time = time.time() - file_start_time
            
            # 记录文件处理信息到日志
            logger.info(f"文件 {uploaded_file.name} 处理完成 - 生成 {len(docs)} 个文档块, 处理耗时: {file_time:.2f}秒")
        
        except Exception as e:
            st.error(f"处理文件 {uploaded_file.name} 时出错: {e}")
            import traceback
            with st.expander("错误详情"):
                st.code(traceback.format_exc())
            continue
    
    # 完成
    progress_bar.progress(1.0)
    status_text.empty()
    total_time = time.time() - total_time
    
    st.success(f"✓ 索引完成！共处理 {len(uploaded_files)} 个文件，生成 {total_docs} 个文档块")
    # 记录总体性能统计到日志
    logger.info(f"索引性能统计 - 处理文件数: {len(uploaded_files)}, 总文档块数: {total_docs}, 总耗时: {total_time:.2f}秒, 平均每个文件: {total_time/len(uploaded_files):.2f}秒")


def main():
    st.title("RAG Chat — Streamlit + Chroma + OpenAI-Compat/Ollama")
    
    # 添加自定义CSS样式
    st.markdown("""
    <style>
    /* 全局重置，避免Streamlit默认样式干扰 */
    body {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        overflow: hidden; /* 防止页面整体滚动 */
    }
    
    /* 重置Streamlit默认容器样式 */
    .stApp {
        background-color: #ffffff;
        padding: 0;
        max-width: 100%;
    }
    
    /* 隐藏Streamlit默认的侧边栏和页头，只保留内容区域 */
    header {
        display: none !important;
    }
    
    /* 调整主内容区域，添加左右边距 */
    .block-container {
        padding: 0 2rem !important;
        margin: 0 auto !important;
        max-width: 1200px !important;
    }
    
    /* 页面布局样式 - 使用flex布局确保输入框在顶部，消息容器占满剩余空间 */
    .main-container {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* 确保输入容器在顶部，并有固定高度 */
    .input-container {
        position: relative;
        background-color: #ffffff;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        width: 100%;
        box-sizing: border-box;
        z-index: 10;
    }
    
    /* 关键修复：使用相对定位确保容器与主页面正确集成，并添加黑色边框 */
    .chat-history-wrapper {
        position: relative; /* 改为相对定位 */
        flex: 1; /* 使用flex布局占满剩余空间 */
        overflow-y: auto; /* 内容超过高度时显示垂直滚动条 */
        border: 2px solid #000000; /* 将边框改为2px黑色实线 */
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #ffffff;
        box-sizing: border-box;
        z-index: 5;
        min-height: 0; /* 允许flex子元素缩小到内容大小 */
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* 确保messages-container完全填充wrapper */
    .messages-container {
        width: 100%;
        padding: 0;
        margin: 0;
    }
    
    /* 确保消息卡片正确显示 */
    .message-card {
        width: 100%;
        margin-bottom: 1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        box-sizing: border-box;
        display: block;
    }
    
    /* 用户消息样式 */
    .user-message {
        background-color: #f0f4f8;
        border-left: 4px solid #1E88E5;
    }
    
    /* 助手消息样式 */
    .assistant-message {
        background-color: #e3f2fd;
        border-left: 4px solid #0D47A1;
    }
    
    /* 问答显示区域样式 */
    .qa-display-area {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 0.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        width: 100%;
        box-sizing: border-box;
    }
    
    /* 来源信息样式 */
    .sources-info {
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f5f5f5;
        border-radius: 0.25rem;
        font-size: 0.875rem;
    }
    
    .user-message {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
    }
    
    .assistant-message {
        background-color: #f8fafc;
        border-left: 4px solid #10b981;
    }
    
    /* 自定义输入框和按钮组合样式 */
    .custom-input-wrapper {
        position: relative;
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .stTextInput > div:first-child {
        width: 100%;
    }
    
    .stTextInput input {
        width: 100%;
        padding-right: 60px; /* 为右侧按钮留出空间 */
        padding-top: 8px;
        padding-bottom: 8px;
        padding-left: 12px;
        border-radius: 8px;
    }
    
    /* 自定义按钮样式 */
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #1565C0;
    }
    
    .stButton > button:active {
        background-color: #0D47A1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 添加Streamlit组件通信的基础脚本
    st.markdown("""
    <script>
        // 确保Streamlit组件API加载完成
        function ensureStreamlitReady() {
            if (window.parent.Streamlit) {
                window.parent.Streamlit.setComponentReady();
                return true;
            }
            return false;
        }
        
        // 初始化时尝试确保Streamlit准备就绪
        ensureStreamlitReady();
        
        // 提供统一的设置session_state值的函数
        function setStreamlitState(key, value) {
            try {
                // 优先使用Streamlit组件API
                if (window.parent.Streamlit && window.parent.Streamlit.setComponentValue) {
                    window.parent.Streamlit.setComponentValue(key, value);
                    return true;
                }
                // 备选方案：使用自定义事件
                if (window.parent && window.parent.document) {
                    window.parent.document.dispatchEvent(
                        new CustomEvent('streamlit:setComponentValue', {
                            detail: {key: key, value: value}
                        })
                    );
                    return true;
                }
                return false;
            } catch (error) {
                console.error('设置Streamlit状态失败:', error);
                return false;
            }
        }
        
        // 暴露给全局使用
        window.setStreamlitState = setStreamlitState;
    </script>
    """, unsafe_allow_html=True)

    # 初始化对话线程和当前线程ID
    if "conversation_threads" not in st.session_state:
        st.session_state.conversation_threads = load_conversation_threads()
    
    if "current_thread_id" not in st.session_state:
        # 如果没有线程，创建一个新线程
        if not st.session_state.conversation_threads:
            thread_id, _ = create_new_conversation_thread(st.session_state.conversation_threads)
            st.session_state.current_thread_id = thread_id
        else:
            # 否则使用最新的线程
            st.session_state.current_thread_id = max(
                st.session_state.conversation_threads.keys(),
                key=lambda k: st.session_state.conversation_threads[k].get('last_updated', '')
            )
    
    # 初始化自定义按钮点击状态
    if 'custom_ask_clicked' not in st.session_state:
        st.session_state.custom_ask_clicked = False
    
    # 初始化删除对话相关状态
    if 'thread_to_delete' not in st.session_state:
        st.session_state.thread_to_delete = None
    if 'show_delete_confirm' not in st.session_state:
        st.session_state.show_delete_confirm = False
    
    # 显示删除确认对话框
    if st.session_state.show_delete_confirm and st.session_state.thread_to_delete:
        with st.sidebar:
            thread_name = st.session_state.conversation_threads.get(
                st.session_state.thread_to_delete, {}).get('name', '未知对话')
            st.error(f"确定要删除对话 '{thread_name}' 吗？此操作无法撤销。")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("确认删除", type="primary", use_container_width=True):
                    # 调用删除函数（稍后实现）
                    delete_conversation_thread(st.session_state.thread_to_delete)
                    # 重置状态
                    st.session_state.thread_to_delete = None
                    st.session_state.show_delete_confirm = False
                    # 如果删除的是当前选中的线程，切换到另一个线程或创建新线程
                    if st.session_state.current_thread_id == st.session_state.thread_to_delete:
                        if st.session_state.conversation_threads:
                            # 切换到最近的线程
                            st.session_state.current_thread_id = next(iter(st.session_state.conversation_threads))
                        else:
                            # 创建新线程
                            st.session_state.current_thread_id = create_new_conversation_thread()
                    # 刷新页面
                    st.rerun()
            with col2:
                if st.button("取消", use_container_width=True):
                    # 重置状态
                    st.session_state.thread_to_delete = None
                    st.session_state.show_delete_confirm = False
                    # 刷新页面
                    st.rerun()

    # 显示 LLM 服务状态和对话线程管理（在侧边栏顶部）
    llm_status = get_llm_status()
    with st.sidebar:
        st.header("LLM 知识库助手")
        
        # 新对话按钮
        if st.button("💬 新对话", type="primary", use_container_width=True):
            thread_id, _ = create_new_conversation_thread(st.session_state.conversation_threads)
            st.session_state.current_thread_id = thread_id
            # 刷新页面以显示新对话
            st.rerun()
        
        # 对话线程列表
        st.subheader("对话历史")
        
        # 对线程按最后更新时间排序（最新的在前）
        sorted_threads = sorted(
            st.session_state.conversation_threads.items(),
            key=lambda x: x[1].get('last_updated', ''),
            reverse=True
        )
        
        # 显示线程列表
        for thread_id, thread in sorted_threads:
            is_active = thread_id == st.session_state.current_thread_id
            button_label = f"{thread['name']}"
            
            # 使用列布局显示线程名称和删除按钮
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
                    st.session_state.current_thread_id = thread_id
                    # 刷新页面以显示选中的对话
                    st.rerun()
            with col2:
                # 添加删除按钮
                if st.button("🗑️", key=f"delete_thread_{thread_id}", use_container_width=True):
                    # 标记要删除的线程ID
                    st.session_state.thread_to_delete = thread_id
                    # 显示确认对话框
                    st.session_state.show_delete_confirm = True
                    # 刷新页面以显示确认对话框
                    st.rerun()
        
        st.divider()
        
        st.header("LLM 服务状态")
        if llm_status["current_service"] == "OpenAI-compatible API":
            st.success(f"✅ **当前使用: {llm_status['current_service']}**")
            st.write(f"模型: {llm_status['current_model']}")
            if llm_status["fallback_service"] != "无":
                st.info(f"回退服务: {llm_status['fallback_service']}")
        elif llm_status["current_service"] == "Ollama":
            st.info(f"ℹ️ **当前使用: {llm_status['current_service']}**")
            st.write(f"模型: {llm_status['current_model']}")
        else:
            st.error(f"❌ **{llm_status['current_service']}**")
            st.warning("请配置 openai-compatible API 或确保 Ollama 正在运行")
        
        # Display configuration details
        with st.expander("📋 配置详情"):
            st.write(f"**openai-compatible:** {'✅ 已配置' if llm_status['openai_configured'] else '❌ 未配置'}")
            if not llm_status['openai_configured']:
                st.write("缺少的配置:")
                if not llm_status.get('openai_api_key_set', False):
                    st.write("  - ❌ OPENAI_COMPATIBLE_API_KEY")
                st.write("")
                st.write("💡 **解决方法:**")
                st.write("1. 在项目根目录创建 `.env` 文件")
                st.write("2. 添加以下配置:")
                st.code("""
OPENAI_COMPATIBLE_API_KEY=你的API密钥
OPENAI_COMPATIBLE_MODEL=<model name>
                """, language="env")
            st.write(f"**Ollama:** {'✅ 可用' if llm_status['ollama_available'] else '❌ 不可用'}")
            st.write(f"**最大对话线程数:** {MAX_CONVERSATION_THREADS}")
        
        st.divider()
    
    st.sidebar.header("Upload")
    st.sidebar.caption("支持上传最多 5 个文件（.txt, .md, .pdf, .docx）")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents to index", 
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf', 'docx']
    )
    
    if uploaded_files:
        # 检查文件数量
        if len(uploaded_files) > 5:
            st.sidebar.warning(f"⚠️ 您选择了 {len(uploaded_files)} 个文件，将只处理前 5 个")
        
        # 显示文件列表
        with st.sidebar.expander(f"📁 已选择文件 ({len(uploaded_files)} 个)"):
            for i, f in enumerate(uploaded_files[:5], 1):
                st.write(f"{i}. {f.name} ({f.size / 1024:.1f} KB)")
        
        if st.sidebar.button("Ingest", type="primary"):
            ingest_and_index(uploaded_files[:5])  # 只处理前5个

    # 输入容器移到顶部
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # 使用 session state 跟踪当前问题，确保每次新问题时都更新
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""
    
    # 使用默认值而不是滑块控件
    k = 4  # 默认检索4个文档片段
    min_similarity = 0.3  # 默认最小相似度阈值30%
    
    # 创建自定义输入框和按钮组合
    st.markdown('<div class="custom-input-wrapper">', unsafe_allow_html=True)
    
    # 使用st.text_input并设置key参数
    question = st.text_input(
        "",
        key="user_question",
        placeholder="您的问题"
    )
    
    # 使用Streamlit官方按钮，通过key和样式类进行自定义
    st.markdown("""
    <style>
    /* 为Streamlit按钮添加自定义样式 */
    .stButton > button {
        display: block;
        width: 100%;
        margin-top: 10px;
        padding: 8px 0;
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #1565C0;
    }
    
    .stButton > button:active {
        background-color: #0D47A1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 使用Streamlit官方的st.button而不是自定义HTML按钮
    ask_button = st.button("Ask", key="custom_ask_button")
    
    # 检查是否是新问题
    is_new_question = question != st.session_state.last_question if question != st.session_state.last_question else True
    
    # 移除自定义的Enter键事件监听器，使用Streamlit原生功能
    
    # 确保侧边栏的Ingest按钮可见（剩余的隐藏样式会在页面底部定义）
    st.markdown("""
    <style>
        # 确保侧边栏的Ingest按钮可见
        .sidebar-content [data-testid="stButton"] {
            display: block !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭custom-input-wrapper
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭input-container
    
    # 添加问答显示区域，位于ask按钮下方
    st.markdown('<div class="qa-display-area">', unsafe_allow_html=True)
    
    # 获取当前线程的对话历史
    current_thread = st.session_state.conversation_threads.get(st.session_state.current_thread_id, {})
    conversation_history = current_thread.get('conversation_history', [])
    
    # 如果对话历史为空，添加一个欢迎消息
    if not conversation_history:
        conversation_history = [{
            "role": "assistant",
            "content": "👋 你好！我是一个基于 LLM 的知识库问答助手。请输入你的问题，我会根据知识库内容为你提供回答。",
            "timestamp": "欢迎消息"
        }]
        # 更新线程信息
        current_thread['conversation_history'] = conversation_history
        current_thread['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.conversation_threads[st.session_state.current_thread_id] = current_thread
        # 保存对话线程
        save_conversation_threads(st.session_state.conversation_threads)
    
    # 显示最新的问题和答案（如果有）
    if conversation_history:
        # 获取最近的对话（问题和答案）
        recent_messages = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        
        for message in recent_messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # 根据角色选择不同的样式类
            message_class = "user-message" if role == "user" else "assistant-message"
            
            # 显示消息内容
            st.markdown(f'<div class="message-card {message_class}">{content}</div>', unsafe_allow_html=True)
            
            # 如果是助手消息且有来源信息，显示来源
            if role == "assistant" and "sources" in message and message["sources"]:
                st.markdown('<div class="sources-info">', unsafe_allow_html=True)
                st.markdown('**来源:**', unsafe_allow_html=True)
                for source in message["sources"]:
                    st.markdown(source, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭qa-display-area
    
    # 检查是否应该处理问题（使用Streamlit原生按钮返回值）
    if ask_button and question:
        # 更新问题记录
        st.session_state.last_question = question
        # 重置自定义按钮点击状态（保留兼容性）
        if hasattr(st.session_state, 'custom_ask_clicked'):
            st.session_state.custom_ask_clicked = False
        
        start_time = time.time()
        
        # Retrieval phase - 每次都会重新检索
        retrieve_start = time.time()
        store = get_store()
        
        # 获取当前线程的对话历史
        current_thread = st.session_state.conversation_threads.get(st.session_state.current_thread_id, {})
        conversation_history = current_thread.get('conversation_history', [])
        
        # 创建增强的查询，包含最近的对话历史上下文
        enhanced_query = question
        recent_conversations = []
        
        # 获取最近的对话历史（最多2轮对话）
        if len(conversation_history) > 2:
            # 从历史中提取最近的对话（跳过当前问题和欢迎消息）
            i = len(conversation_history) - 1
            while i >= 0 and len(recent_conversations) < 4:  # 最多4条消息（2轮对话）
                if conversation_history[i].get('role') in ['user', 'assistant'] and \
                   conversation_history[i].get('timestamp') != '欢迎消息':
                    recent_conversations.append(conversation_history[i])
                i -= 1
            
            # 将最近的对话按时间顺序排列（最早的在前）
        recent_conversations.reverse()
        
        # 构建增强的查询，包含最近的对话历史
        conversation_context = "\n".join([
            f"{'用户' if msg.get('role') == 'user' else 'AI'}: {msg.get('content', '')}"
            for msg in recent_conversations[-4:]  # 最多使用最近4条消息
        ])
        
        if conversation_context:
            enhanced_query = f"{question}\n\n最近的对话上下文:\n{conversation_context}"
            logger.info("使用增强查询，包含最近对话历史")
        
        # 先检索原始结果（不应用阈值）用于调试
        raw_hits = store.query(enhanced_query, k=k * 5, min_similarity=0.0)
        
        # 然后应用阈值过滤
        hits = store.query(enhanced_query, k=k, min_similarity=min_similarity)
        retrieve_time = time.time() - retrieve_start
        
        # 记录检索结果信息到终端日志
        if not hits:
            logger.warning(f"没有检索到任何相关内容！当前最小相似度阈值: {min_similarity:.0%}")
            if raw_hits:
                logger.info(f"原始检索结果（未过滤，共 {len(raw_hits)} 个）:")
                for i, h in enumerate(raw_hits[:5], 1):
                    text = h.get('document') or h.get('text') or ""
                    metadata = h.get('metadata', {})
                    similarity = h.get('similarity', 0)
                    source = metadata.get('source', '未知')
                    source_type = "对话历史" if source == "conversation_history" else "文档"
                    logger.info(f"原始片段 {i}: 相似度: {similarity:.2%}, 来源类型: {source_type}, 来源: {source}")
        else:
            logger.info(f"检索到 {len(hits)} 个片段（已过滤相似度 < {min_similarity:.0%} 的片段）:")
            for i, h in enumerate(hits, 1):
                text = h.get('document') or h.get('text') or ""
                metadata = h.get('metadata', {})
                similarity = h.get('similarity', 0)
                source = metadata.get('source', '未知')
                source_type = "对话历史" if source == "conversation_history" else "文档"
                logger.info(f"片段 {i}: 相似度: {similarity:.2%}, 来源类型: {source_type}, 来源: {source}, 内容长度: {len(text)} 字符")
        
        # 构建提示词，包含对话历史上下文
        prompt_start = time.time()
        
        # 准备对话历史上下文
        conversation_history_context = ""
        if len(conversation_history) > 2:
            # 获取最近的对话历史（最多3轮对话）
            recent_history = []
            i = len(conversation_history) - 1  # 跳过当前问题
            while i >= 0 and len(recent_history) < 6:  # 最多6条消息（3轮对话）
                if conversation_history[i].get('role') in ['user', 'assistant'] and \
                   conversation_history[i].get('timestamp') != '欢迎消息':
                    recent_history.append(conversation_history[i])
                i -= 1
            
            # 按时间顺序排列
            recent_history.reverse()
            
            # 构建对话历史上下文
            conversation_history_context = "\n\n对话历史上下文:\n" + "\n".join([
                f"{'用户' if msg.get('role') == 'user' else 'AI'}: {msg.get('content', '')}"
                for msg in recent_history[-6:]
            ])
        
        if not hits:
            system_prompt = """你是一个有帮助的AI助手。
你的任务是回答用户的问题。
请考虑对话历史上下文，尽可能连贯地回应用户的问题。
如果没有足够的上下文信息，请礼貌地告诉用户。
"""
            prompt = f"""{system_prompt}
{conversation_history_context}

用户问题: {question}
"""
        else:
            system_prompt = """你是一个有帮助的AI助手。
你的任务是基于提供的上下文信息和对话历史回答用户的问题。
请使用中文回答，语言要自然、友好。
请考虑对话的连贯性，参考之前的对话内容。
请尽可能简洁地回答，不要做过多的额外扩展。
请使用提供的上下文信息，不要编造不存在的内容。
如果上下文信息不足以回答问题，请如实说明。
请在回答的最后使用 <sources> 标签列出你参考的内容来源。
"""
            
            # 构建上下文
            context = "\n\n".join([
                f"片段 {i + 1}:\n{h.get('document') or h.get('text') or ''}" 
                for i, h in enumerate(hits)
            ])
            
            prompt = f"""{system_prompt}

上下文信息:
{context}
{conversation_history_context}

用户问题: {question}
"""
        
        prompt_time = time.time() - prompt_start
        
        # 记录prompt和检索信息到终端日志
        logger.info(f"Prompt 长度: {len(prompt)} 字符 | 检索耗时: {retrieve_time:.2f}秒 | 检索到 {len(hits)} 个片段（阈值: {min_similarity:.0%})")
        logger.debug(f"完整Prompt: {prompt}")
        
        # LLM generation phase
        with st.spinner("🤖 正在生成答案..."):
            generate_start = time.time()
            try:
                answer, service_used = generate(prompt)
                generate_time = time.time() - generate_start
                total_time = time.time() - start_time
                
                # 记录性能统计信息到终端
                logger.info(f"性能统计 - 向量检索: {retrieve_time:.2f}秒, Prompt构建: {prompt_time:.2f}秒, LLM生成: {generate_time:.2f}秒 ({service_used}), 总耗时: {total_time:.2f}秒")
                
                # 获取当前时间戳
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                
                    # 获取当前线程
                current_thread = st.session_state.conversation_threads.get(st.session_state.current_thread_id, {})
                conversation_history = current_thread.get('conversation_history', [])
                
                # 检查是否是线程的第一个问题，如果是则更新线程名称
                if len(conversation_history) == 1 and conversation_history[0].get('role') == 'assistant':
                    update_thread_name(st.session_state.conversation_threads, st.session_state.current_thread_id, question)
                
                # 将问题和答案添加到对话历史
                conversation_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": current_time
                })
                
                # 准备来源信息
                sources = []
                for h in hits:
                    metadata = h.get("metadata", {})
                    similarity = h.get('similarity', 0)
                    sources.append(f"- {metadata.get('source', '未知')} (块 {metadata.get('chunk_index', '?')}, 相似度: {similarity:.2%})")
                
                conversation_history.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "timestamp": current_time
                })
                
                # 更新线程信息
                current_thread['conversation_history'] = conversation_history
                current_thread['last_updated'] = current_time
                st.session_state.conversation_threads[st.session_state.current_thread_id] = current_thread
                
                # 保存对话线程到本地文件
                save_conversation_threads(st.session_state.conversation_threads)
                
                # 也保存到传统的对话历史文件（保持向后兼容）
                save_conversation_history(conversation_history)
                
                # 刷新页面以显示新的对话内容
                st.rerun()
                
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                st.error(f"LLM generation error: {e}")



if __name__ == "__main__":
    main()
