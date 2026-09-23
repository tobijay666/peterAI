"""Interactive REPL engine loop for user interaction with Peter."""

import sys
from typing import Optional
from agents.peter import PeterAgent
from core.console import Colors, format_agent, format_error, format_prompt, format_warn
from core.guardrails import GuardrailManager
from core.logger import logger
from tts.service import TTSService


class REPLEngine:
    """Manages the interactive terminal REPL session for Sir Anton."""

    def __init__(self, peter_agent: PeterAgent, tts_service: TTSService, guardrails: GuardrailManager):
        self.peter = peter_agent
        self.tts = tts_service
        self.guardrails = guardrails
        self.running = False

    def start(self) -> None:
        self.running = True
        print(f"\n{Colors.CYAN}{Colors.BOLD}===================================================={Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}     Peter AI Multi-Agent Harness (v1.0.0)          {Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}===================================================={Colors.RESET}\n")

        # Startup Announcement
        welcome_msg = "Peter online. Ready to assist, Sir."
        print(format_agent("Peter", welcome_msg))
        self.tts.speak(welcome_msg)

        while self.running:
            try:
                user_input = input(format_prompt("\nSir Anton > ")).strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print(format_agent("Peter", "Shutting down system. Goodbye, Sir."))
                    self.tts.speak_blocking("Peter going offline. Goodbye, Sir.")
                    self.running = False
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if user_input.lower() == "status":
                    self._show_status()
                    continue

                if user_input.lower() == "reload-guardrails":
                    self.guardrails.reload()
                    print(format_agent("Peter", "Guardrail rules reloaded successfully, Sir."))
                    continue

                # Run Peter agent
                response = self.peter.run(user_input=user_input)

                # Output final response
                print("\n" + format_agent("Peter", response))
                self.tts.speak(response)

            except KeyboardInterrupt:
                print("\n" + format_warn("Interrupt signal received. Type 'exit' to quit."))
            except Exception as e:
                logger.error(f"REPL Loop Exception: {e}")
                print(format_error(f"An error occurred: {e}"))

    def _show_help(self) -> None:
        print("\n--- Peter AI Help Commands ---")
        print(" exit / quit  : Shut down the system")
        print(" help         : Display command instructions")
        print(" status       : Display agent state and provider token usage")
        print(" reload-guardrails : Reload security rules from config/guardrails.json")

    def _show_status(self) -> None:
        print("\n--- System Status ---")
        print(f"Peter State : {self.peter.state.value}")
        print(f"Specialists : {list(self.peter.specialists.keys())}")
        print(f"Token Stats : {self.peter.router.get_token_usage()}")
