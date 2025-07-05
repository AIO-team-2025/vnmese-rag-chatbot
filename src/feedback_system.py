import streamlit as st
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class FeedbackSystem:
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self.feedback_data = self._load_feedback_data()
    
    def _load_feedback_data(self) -> List[Dict]:
        """Tải dữ liệu feedback từ file"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu feedback: {e}")
            return []
    
    def _save_feedback_data(self):
        """Lưu dữ liệu feedback vào file"""
        try:
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Lỗi khi lưu dữ liệu feedback: {e}")
    
    def add_feedback(self, question: str, answer: str, rating: str, 
                    feedback_text: str = "", pdf_name: str = ""):
        """Thêm feedback mới"""
        feedback_entry = {
            "id": len(self.feedback_data) + 1,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "rating": rating,  # "like" hoặc "dislike"
            "feedback_text": feedback_text,
            "pdf_name": pdf_name
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback_data()
        return feedback_entry["id"]
    
    def get_feedback_stats(self) -> Dict:
        """Lấy thống kê feedback"""
        if not self.feedback_data:
            return {"total": 0, "likes": 0, "dislikes": 0, "like_rate": 0}
        
        total = len(self.feedback_data)
        likes = sum(1 for f in self.feedback_data if f["rating"] == "like")
        dislikes = total - likes
        like_rate = (likes / total) * 100 if total > 0 else 0
        
        return {
            "total": total,
            "likes": likes,
            "dislikes": dislikes,
            "like_rate": round(like_rate, 1)
        }
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Lấy feedback gần đây"""
        return sorted(self.feedback_data, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def display_feedback_form(self, question: str, answer: str, 
                            message_id: str, pdf_name: str = ""):
        """Hiển thị form feedback cho một câu trả lời"""
        
        # Tạo key duy nhất cho mỗi message
        like_key = f"like_{message_id}"
        dislike_key = f"dislike_{message_id}"
        feedback_key = f"feedback_{message_id}"
        
        # Container cho buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        
        with col1:
            if st.button("👍", key=like_key, help="Thích câu trả lời này"):
                self._handle_rating(question, answer, "like", message_id, pdf_name)
        
        with col2:
            if st.button("👎", key=dislike_key, help="Không thích câu trả lời này"):
                self._handle_rating(question, answer, "dislike", message_id, pdf_name)
        
        # Hiển thị form feedback nếu đã có rating
        if f"rated_{message_id}" in st.session_state:
            rating = st.session_state[f"rated_{message_id}"]
            
            with st.expander("📝 Để lại feedback chi tiết", expanded=True):
                feedback_text = st.text_area(
                    "Nhận xét của bạn:",
                    placeholder="Hãy cho biết tại sao bạn thích/không thích câu trả lời này...",
                    key=f"feedback_text_{message_id}",
                    height=100
                )
                
                col_submit, col_skip = st.columns([1, 1])
                
                with col_submit:
                    if st.button("✅ Gửi feedback", key=f"submit_{message_id}"):
                        feedback_id = self.add_feedback(
                            question, answer, rating, feedback_text, pdf_name
                        )
                        st.success(f"🎉 Cảm ơn bạn đã feedback! (ID: {feedback_id})")
                        # Xóa trạng thái để không hiển thị form nữa
                        del st.session_state[f"rated_{message_id}"]
                        st.rerun()
                
                with col_skip:
                    if st.button("⏭️ Bỏ qua", key=f"skip_{message_id}"):
                        feedback_id = self.add_feedback(
                            question, answer, rating, "", pdf_name
                        )
                        st.info(f"Đã lưu đánh giá của bạn! (ID: {feedback_id})")
                        del st.session_state[f"rated_{message_id}"]
                        st.rerun()
    
    def _handle_rating(self, question: str, answer: str, rating: str, 
                      message_id: str, pdf_name: str = ""):
        """Xử lý khi người dùng chọn like/dislike"""
        st.session_state[f"rated_{message_id}"] = rating
        
        # Hiển thị thông báo tạm thời
        if rating == "like":
            st.success("👍 Cảm ơn bạn đã thích câu trả lời!")
        else:
            st.info("👎 Cảm ơn bạn đã đánh giá!")
        
        st.rerun()
    
    def display_feedback_analytics(self):
        """Hiển thị thống kê feedback trong sidebar"""
        st.subheader("📊 Thống kê Feedback")
        
        stats = self.get_feedback_stats()
        
        if stats["total"] > 0:
            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tổng số", stats["total"])
                st.metric("👍 Thích", stats["likes"])
            with col2:
                st.metric("Tỷ lệ hài lòng", f"{stats['like_rate']}%")
                st.metric("👎 Không thích", stats["dislikes"])
            
            # Progress bar
            st.progress(stats["like_rate"] / 100)
            
            # Recent feedback
            if st.checkbox("Xem feedback gần đây"):
                recent = self.get_recent_feedback(5)
                for i, feedback in enumerate(recent, 1):
                    with st.expander(f"Feedback #{i} - {feedback['rating']}"):
                        st.write(f"**Q:** {feedback['question'][:100]}...")
                        st.write(f"**A:** {feedback['answer'][:100]}...")
                        if feedback['feedback_text']:
                            st.write(f"**Nhận xét:** {feedback['feedback_text']}")
                        st.write(f"**Thời gian:** {feedback['timestamp']}")
        else:
            st.info("Chưa có feedback nào!")
    
    def export_feedback_data(self) -> str:
        """Xuất dữ liệu feedback ra CSV"""
        if not self.feedback_data:
            return None
        
        df = pd.DataFrame(self.feedback_data)
        csv_file = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        return csv_file
    
    def get_feedback_for_training(self) -> List[Dict]:
        """Lấy dữ liệu feedback để training model"""
        training_data = []
        
        for feedback in self.feedback_data:
            training_item = {
                "input": feedback["question"],
                "output": feedback["answer"],
                "rating": 1 if feedback["rating"] == "like" else 0,
                "feedback": feedback["feedback_text"],
                "context": feedback["pdf_name"]
            }
            training_data.append(training_item)
        
        return training_data
    
    def clear_feedback_data(self):
        """Xóa toàn bộ dữ liệu feedback"""
        self.feedback_data = []
        self._save_feedback_data()

# Singleton instance
_feedback_system = None

def get_feedback_system() -> FeedbackSystem:
    """Lấy instance của FeedbackSystem"""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = FeedbackSystem()
    return _feedback_system

def display_message_with_feedback(role: str, content: str, question: str = "", 
                                pdf_name: str = "", message_id: str = ""):
    """Hiển thị message với tùy chọn feedback"""
    with st.chat_message(role):
        st.write(content)
        
        # Chỉ hiển thị feedback cho assistant
        if role == "assistant" and question and message_id:
            feedback_system = get_feedback_system()
            feedback_system.display_feedback_form(
                question, content, message_id, pdf_name
            )

def display_feedback_sidebar():
    """Hiển thị phần feedback trong sidebar"""
    feedback_system = get_feedback_system()
    
    with st.sidebar:
        st.markdown("---")
        feedback_system.display_feedback_analytics()
        
        # Export button
        if st.button("📁 Xuất dữ liệu feedback"):
            csv_file = feedback_system.export_feedback_data()
            if csv_file:
                st.success(f"Đã xuất: {csv_file}")
            else:
                st.info("Không có dữ liệu để xuất!")
        
        # Clear button (with confirmation)
        if st.button("🗑️ Xóa tất cả feedback"):
            if st.checkbox("Xác nhận xóa"):
                feedback_system.clear_feedback_data()
                st.success("Đã xóa toàn bộ feedback!")
                st.rerun()
