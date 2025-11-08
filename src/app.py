"""Streamlit app: upload files, ingest into Chroma, ask queries."""
import streamlit as st
from dotenv import load_dotenv
import os
import uuid
import time

from ingest import save_uploaded_file, extract_text_from_file, chunk_text
from vector_store import VectorStore
from retriever import build_prompt
from llm_adapter import generate, get_llm_status

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


@st.cache_resource
def get_store():
    return VectorStore(persist_directory=CHROMA_DIR)


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
    st.info("🗑️ 已清除上一次上传的所有文档")
    
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
            
            # 显示单个文件处理结果
            with st.expander(f"📄 {uploaded_file.name} ({len(docs)} 个文档块)"):
                st.write(f"- 提取文本长度: {len(text)} 字符")
                st.write(f"- 文本提取: {extract_time:.2f}秒")
                st.write(f"- 文本分块: {chunk_time:.2f}秒")
                st.write(f"- Embedding 生成: {embed_time:.2f}秒")
                st.write(f"- 文件处理耗时: {file_time:.2f}秒")
                # 显示文本预览
                st.write(f"- 文本预览: {text[:300]}...")
        
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
    with st.expander("⏱️ 总体性能统计"):
        st.write(f"- 处理文件数: {len(uploaded_files)}")
        st.write(f"- 总文档块数: {total_docs}")
        st.write(f"- **总耗时: {total_time:.2f}秒**")
        st.write(f"- 平均每个文件: {total_time/len(uploaded_files):.2f}秒")


def main():
    st.title("RAG Chat — Streamlit + Chroma + OpenAI-Compat/Ollama")
    
    # 显示 LLM 服务状态（在侧边栏顶部）
    llm_status = get_llm_status()
    with st.sidebar:
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

    st.header("Ask a question")
    question = st.text_input("Your question")
    
    # 使用 session state 跟踪当前问题，确保每次新问题时都更新
    if 'last_question' not in st.session_state:
        st.session_state.last_question = ""
    
    # 检查是否是新问题
    is_new_question = question != st.session_state.last_question
    
    col1, col2 = st.columns(2)
    with col1:
        k = st.slider("retrieval k", 1, 10, 4, help="检索的文档片段数量")
    with col2:
        min_similarity = st.slider(
            "最小相似度", 
            0.0, 1.0, 0.3, 0.05,
            help="相似度低于此值的片段将被过滤（0.3 表示 30% 相似度）"
        )

    if st.button("Ask") and question:
        # 更新问题记录
        st.session_state.last_question = question
        
        start_time = time.time()
        
        # Retrieval phase - 每次都会重新检索
        retrieve_start = time.time()
        store = get_store()
        
        # 先检索原始结果（不应用阈值）用于调试
        raw_hits = store.query(question, k=k * 5, min_similarity=None)
        
        # 然后应用阈值过滤
        hits = store.query(question, k=k, min_similarity=min_similarity)
        retrieve_time = time.time() - retrieve_start
        
        # 调试：显示检索到的上下文（每次提问都会更新）
        with st.expander("🔍 检索到的上下文（调试）", expanded=True):
            if not hits:
                st.warning("⚠️ 没有检索到任何相关内容！")
                st.info(f"提示：尝试降低最小相似度阈值（当前: {min_similarity:.0%}）")
                
                # 显示未过滤的原始结果，帮助用户了解距离分布
                if raw_hits:
                    st.write("---")
                    st.write(f"**原始检索结果（未过滤，共 {len(raw_hits)} 个）：**")
                    for i, h in enumerate(raw_hits[:5], 1):  # 只显示前5个
                        text = h.get('document') or h.get('text') or ""
                        metadata = h.get('metadata', {})
                        distance = h.get('distance', 0)
                        similarity = h.get('similarity', 0)
                        
                        st.write(f"**原始片段 {i}:**")
                        st.write(f"- 相似度: **{similarity:.2%}** (距离: {distance:.4f})")
                        st.write(f"- 来源: {metadata.get('source', '未知')}")
                        st.write(f"- 内容预览: {text[:150]}...")
                        st.write("---")
            else:
                threshold_info = f"（已过滤相似度 < {min_similarity:.0%} 的片段）"
                st.write(f"检索到 {len(hits)} 个片段{threshold_info}：")
                for i, h in enumerate(hits, 1):
                    st.write(f"**片段 {i}:**")
                    text = h.get('document') or h.get('text') or ""
                    metadata = h.get('metadata', {})
                    distance = h.get('distance', 0)
                    similarity = h.get('similarity', 0)
                    
                    # 根据相似度显示不同的颜色
                    if similarity >= 0.7:
                        similarity_color = "🟢"
                    elif similarity >= 0.5:
                        similarity_color = "🟡"
                    else:
                        similarity_color = "🟠"
                    
                    st.write(f"- {similarity_color} 相似度: **{similarity:.2%}** (距离: {distance:.4f})")
                    st.write(f"- 来源: {metadata.get('source', '未知')}")
                    st.write(f"- 内容长度: {len(text)} 字符")
                    st.write(f"- 内容预览: {text[:200]}...")
                    st.write("---")
        
        prompt_start = time.time()
        prompt = build_prompt(question, hits)
        prompt_time = time.time() - prompt_start
        
        # 调试：显示完整的 prompt（每次提问都会更新）
        with st.expander("📝 完整 Prompt（调试）", expanded=False):
            st.code(prompt, language=None)
        
        st.info(f"📊 Prompt 长度: {len(prompt)} 字符 | 检索耗时: {retrieve_time:.2f}秒 | 检索到 {len(hits)} 个片段（阈值: {min_similarity:.0%}）")
        
        # LLM generation phase
        with st.spinner("🤖 正在生成答案..."):
            generate_start = time.time()
            try:
                answer, service_used = generate(prompt)
                generate_time = time.time() - generate_start
                total_time = time.time() - start_time
                
                # 显示使用的服务
                if service_used == "openai-compatible (Moonshot AI)":
                    st.success(f"✅ 使用服务: {service_used}")
                elif "Ollama" in service_used:
                    st.info(f"ℹ️ 使用服务: {service_used}")
                
                st.markdown("**Answer**")
                st.write(answer)
                
                st.markdown("**Sources**")
                for h in hits:
                    metadata = h.get("metadata", {})
                    similarity = h.get('similarity', 0)
                    st.write(f"- {metadata.get('source', '未知')} (块 {metadata.get('chunk_index', '?')}, 相似度: {similarity:.2%})")
                
                # 显示性能统计
                with st.expander("⏱️ 性能统计"):
                    st.write(f"- 向量检索: {retrieve_time:.2f}秒")
                    st.write(f"- Prompt 构建: {prompt_time:.2f}秒")
                    st.write(f"- **LLM 生成: {generate_time:.2f}秒** ({service_used})")
                    st.write(f"- **总耗时: {total_time:.2f}秒**")
                    
            except Exception as e:
                st.error(f"LLM generation error: {e}")


if __name__ == "__main__":
    main()
