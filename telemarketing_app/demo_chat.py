from telemarketing import TelemarketingEngine


def main() -> None:
    engine = TelemarketingEngine()
    print("Telemarketing demo started. 输入 exit 退出。")
    while True:
        user_text = input("客户: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        result = engine.chat(
            user_text=user_text,
            external_user_id="cli-user",
            session_key="cli-session",
            channel="phone",
            nickname="演示客户",
        )
        print(f"顾问: {result['reply']}")
        print(f"  intent={result['intent']} reply_type={result['reply_type']} next_action={result['next_action']}")


if __name__ == "__main__":
    main()
