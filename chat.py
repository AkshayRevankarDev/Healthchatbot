"""
Interactive mental health screening chat — runs in your terminal.
"""
import sys
from dialogue_manager import build_dialogue_graph, create_initial_state, run_interactive_turn

def main():
    print("=" * 60)
    print("Mental Health Screening — Interactive Session")
    print("Type 'quit' to exit.")
    print("=" * 60)

    graph = build_dialogue_graph()
    state = create_initial_state()

    opening = "Hi, thanks for coming in today. How have things been going for you lately?"
    print(f"\nTherapist: {opening}\n")
    state["messages"].append({"role": "assistant", "content": opening})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("[Session ended]")
            break

        state, response = run_interactive_turn(user_input, state, graph)
        print(f"\nTherapist: {response}\n")

        if state.get("session_complete") or state.get("safety_triggered"):
            print("─" * 60)
            if state.get("session_complete"):
                scores = state.get("scores", {})
                total = sum(scores.values())
                print(f"Session complete. Scores: {scores}  (Total PHQ-9: {total})")
            break

if __name__ == "__main__":
    main()
