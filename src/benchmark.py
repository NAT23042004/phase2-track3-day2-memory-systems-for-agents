from dotenv import load_dotenv

from src.agent import MultiMemoryAgent

load_dotenv()


class BenchmarkRunner:
    def __init__(self, log_file="benchmark_detailed_results.txt"):
        self.agent = MultiMemoryAgent()
        self.log_file = log_file
        # Xóa file log cũ nếu có
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== NHẬT KÝ CHI TIẾT BENCHMARK MULTI-MEMORY AGENT (TIẾNG VIỆT) ===\n\n")

    def log(self, message: str):
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def run_scenario(self, name: str, turns: list[str]):
        self.log(f"--- Kịch bản: {name} ---")
        self.agent.short_term.clear()

        for i, query in enumerate(turns):
            self.log(f"Lượt {i + 1} Người dùng: {query}")
            result = self.agent.run(query)
            self.log(f"Lượt {i + 1} Agent (Ý định: {result['intent']}): {result['response']}\n")
        self.log("-" * 30 + "\n")


def run_benchmarks():
    runner = BenchmarkRunner()

    # 1. Profile Recall (Ghi nhớ thông tin cá nhân)
    runner.run_scenario("Ghi nhớ Profile", ["Chào bạn, tôi tên là Natus.", "Tên của tôi là gì?"])

    # 2. Conflict Update (Cập nhật xung đột thông tin - QUAN TRỌNG)
    runner.run_scenario(
        "Cập nhật xung đột",
        [
            "Tôi bị dị ứng với sữa bò.",
            "À tôi nhầm, tôi bị dị ứng đậu nành chứ không phải sữa bò.",
            "Tôi bị dị ứng với cái gì?",
        ],
    )

    # 3. Episodic Recall (Ghi nhớ trải nghiệm)
    runner.run_scenario(
        "Ghi nhớ trải nghiệm",
        [
            "Hôm nay tôi đang gặp khó khăn khi cài đặt Docker. Container không thể khởi động được.",
            "Bạn có thể giúp tôi sửa lỗi mà tôi đã nhắc tới lúc nãy không?",
        ],
    )

    # 4. Semantic Retrieval (Truy xuất kiến thức)
    # Lưu sẵn kiến thức vào ChromaDB
    runner.agent.semantic.save(
        {
            "text": "Chính sách công ty quy định nhân viên được nghỉ phép 20 ngày mỗi năm.",
            "metadata": {"source": "policy_doc"},
        }
    )
    runner.run_scenario("Truy xuất kiến thức", ["Theo quy định công ty thì tôi được nghỉ phép bao nhiêu ngày một năm?"])

    # 5. Context Trimming (Quản lý Context Window dài)
    long_history_scenario = ["Hãy nói về chủ đề ngẫu nhiên số " + str(i) for i in range(15)]
    long_history_scenario.append("Điều đầu tiên mà chúng ta đã nói trong kịch bản này là gì?")
    runner.run_scenario("Cắt tỉa Context", long_history_scenario)

    # 6. Combined Preference & Logic (Kết hợp sở thích và logic)
    runner.run_scenario(
        "Logic sở thích",
        [
            "Tôi thích sử dụng chế độ tối (dark mode) trên giao diện.",
            "Hãy gợi ý cho tôi một bộ màu cho IDE dựa trên sở thích của tôi.",
        ],
    )

    # 7. Cross-Session Consistency (Tính nhất quán qua các phiên)
    runner.run_scenario("Nhất quán xuyên suốt", ["Bạn còn nhớ tên tôi ở kịch bản đầu tiên là gì không?"])

    # 8. Complex Experience (Trải nghiệm phức tạp)
    runner.run_scenario(
        "Trải nghiệm phức tạp",
        [
            "Tuần trước tôi đã rất vất vất vả để hiểu về cơ chế GIL trong Python.",
            "Hãy tóm tắt những khó khăn kỹ thuật mà tôi gặp phải gần đây.",
        ],
    )

    # 9. Knowledge Discovery (Khám phá kiến thức)
    runner.run_scenario(
        "Khám phá kiến thức",
        ["Tôi muốn biết về thủ đô của nước Pháp.", "Hãy cho tôi biết một sự thật thú vị về thành phố đó."],
    )

    # 10. Final Summary (Tóm tắt cuối cùng)
    runner.run_scenario(
        "Tóm tắt tổng kết",
        ["Hãy tóm tắt tất cả những gì bạn biết về sở thích và những khó khăn của tôi trong ngày hôm nay."],
    )


if __name__ == "__main__":
    # Làm sạch bộ nhớ cho lần chạy mới
    agent = MultiMemoryAgent()
    agent.long_term.clear()
    agent.episodic.clear()
    agent.semantic.clear()

    run_benchmarks()
