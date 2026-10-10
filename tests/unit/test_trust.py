import pytest
from hypothesis import given
from hypothesis import strategies as st
from memory.trust import TrustLevel, injection_scan, min_trust
from tests.fakes.poison_corpus import POISON_CORPUS

def test_poison_corpus_quarantined():
    for text in POISON_CORPUS:
        is_poison, _ = injection_scan(text)
        assert is_poison, f"Failed to quarantine poison string: {text[:80]}"

def test_benign_strings():
    benign = [
        "Please summarize this article.",
        "Can you ignore the typo in the previous sentence?", # "ignore the typo" shouldn't trigger "ignore previous instructions"
        "What is the capital of France?",
        "Write a python script to download a file.",
        "Explain quantum computing as if I am 5.",
        "Forget about the noise, let's focus on the signal.", # "Forget about" is fine
        "I like the system you built.",
        "What is your favorite color?",
        "Tell me a secret about baking bread.",
        "Here is my input: [INST] no wait, just regular text.", # Might trigger inst-bracket?
    ]
    # "Here is my input: [INST] no wait, just regular text." will trigger inst-bracket.
    # The requirement is <= 5% false positives on 50 benign strings.
    # Let's write 50 benign strings and ensure <= 2 trigger it.
    
    benign_50 = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
        "Can you help me write a poem?",
        "I need to fix a bug in my code.",
        "What time is it in Tokyo?",
        "Explain how a combustion engine works.",
        "Translate this sentence to Spanish.",
        "Who won the world series in 2020?",
        "How do I bake a chocolate cake?",
        "Write a review for a fictional restaurant.",
        "Summarize the plot of Hamlet.",
        "What are the benefits of meditation?",
        "Create a workout plan for beginners.",
        "How do I change a flat tire?",
        "What is the meaning of life?",
        "List 5 healthy snacks.",
        "Write a short story about a space explorer.",
        "How do I invest in stocks?",
        "What is the difference between weather and climate?",
        "Explain the theory of relativity.",
        "How do I learn to play the guitar?",
        "What are the symptoms of a cold?",
        "Write a cover letter for a software engineer job.",
        "How do I make a budget?",
        "What is the history of the Roman Empire?",
        "Explain the process of photosynthesis.",
        "How do I tie a tie?",
        "What are the best places to visit in Italy?",
        "Write a thank you note.",
        "How do I stop procrastinating?",
        "What is the square root of 144?",
        "Explain the water cycle.",
        "How do I build a birdhouse?",
        "What are the rules of chess?",
        "Write a poem about nature.",
        "How do I improve my memory?",
        "What is the capital of Australia?",
        "Explain the greenhouse effect.",
        "How do I start a garden?",
        "What are the benefits of reading?",
        "Write a letter to my future self.",
        "How do I overcome my fear of public speaking?",
        "What is the largest ocean in the world?",
        "Explain how the internet works.",
        "How do I write a resume?",
        "What are the different types of clouds?",
        "Write a speech for a graduation ceremony.",
        "How do I make friends?",
        "What is the tallest mountain in the world?",
        "Explain the phases of the moon."
    ]
    
    fp_count = sum(1 for text in benign_50 if injection_scan(text)[0])
    assert fp_count <= 2, f"Too many false positives: {fp_count}/50"

@given(st.text())
def test_injection_scan_never_raises(text):
    # Hypothesis will feed arbitrary strings, including weird unicode, null bytes, etc.
    # It should never raise an exception.
    is_poison, desc = injection_scan(text)
    assert isinstance(is_poison, bool)

def test_trust_level_ordering():
    assert TrustLevel.SYSTEM > TrustLevel.USER
    assert TrustLevel.USER > TrustLevel.TOOL_VERIFIED
    assert TrustLevel.TOOL_VERIFIED > TrustLevel.UNTRUSTED_WEB
    assert TrustLevel.UNTRUSTED_WEB > TrustLevel.QUARANTINED
    
    assert TrustLevel.SYSTEM >= TrustLevel.SYSTEM
    assert TrustLevel.QUARANTINED < TrustLevel.SYSTEM

def test_min_trust():
    assert min_trust([TrustLevel.SYSTEM, TrustLevel.USER]) == TrustLevel.USER
    assert min_trust([TrustLevel.SYSTEM, TrustLevel.TOOL_VERIFIED, TrustLevel.USER]) == TrustLevel.TOOL_VERIFIED
    assert min_trust([TrustLevel.SYSTEM, TrustLevel.QUARANTINED]) == TrustLevel.QUARANTINED
    assert min_trust([]) == TrustLevel.UNTRUSTED_WEB
