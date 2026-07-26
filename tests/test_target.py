import asyncio
import types as pytypes
import pytest
import target

def test_build_openai_messages_prepends_system():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    out = target.build_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1:] == msgs

def test_build_gemini_contents_maps_roles():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    out = target.build_gemini_contents(msgs)
    # assistant maps to "model"; order preserved; text carried through
    assert [c.role for c in out] == ["user", "model"]
    assert out[0].parts[0].text == "hi"

async def test_temperature_reaches_gemini(monkeypatch):
    captured = {}
    monkeypatch.setattr(target, "get_gemini_client", lambda: object())
    async def fake_gemini(client, model, contents, gen_config):
        captured["temp"] = gen_config.temperature
        return pytypes.SimpleNamespace(text="ok")
    monkeypatch.setattr(target, "_gemini_generate", fake_gemini)
    await target.query_target_conversation(
        [{"role": "user", "content": "hi"}], provider="gemini", model="m", temperature=0.7)
    assert captured["temp"] == 0.7

async def test_temperature_reaches_openai(monkeypatch):
    captured = {}
    monkeypatch.setattr(target, "get_openai_client", lambda: object())
    async def fake_openai(client, model, messages, temperature):
        captured["temp"] = temperature
        msg = pytypes.SimpleNamespace(content="ok")
        return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])
    monkeypatch.setattr(target, "_openai_chat", fake_openai)
    await target.query_target_conversation(
        [{"role": "user", "content": "hi"}], provider="openai", model="m", temperature=0.5)
    assert captured["temp"] == 0.5

async def test_temperature_reaches_ollama(monkeypatch):
    captured = {}
    monkeypatch.setattr(target, "get_ollama_client", lambda: object())
    async def fake_openai(client, model, messages, temperature):
        captured["temp"] = temperature
        msg = pytypes.SimpleNamespace(content="ok")
        return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])
    monkeypatch.setattr(target, "_openai_chat", fake_openai)
    await target.query_target_conversation(
        [{"role": "user", "content": "hi"}], provider="ollama", model="m", temperature=0.3)
    assert captured["temp"] == 0.3


async def test_no_system_prompt_omits_openai_system_message(monkeypatch):
    captured = {}
    monkeypatch.setattr(target, "get_openai_client", lambda: object())

    async def fake_openai(client, model, messages, temperature):
        captured["messages"] = messages
        msg = pytypes.SimpleNamespace(content="ok")
        return pytypes.SimpleNamespace(choices=[pytypes.SimpleNamespace(message=msg)])

    monkeypatch.setattr(target, "_openai_chat", fake_openai)
    await target.query_target_conversation(
        [{"role": "user", "content": "hi"}], provider="openai", model="m",
        system_prompt=None)
    assert all(m["role"] != "system" for m in captured["messages"])


async def test_no_system_prompt_omits_gemini_system_instruction(monkeypatch):
    captured = {}
    monkeypatch.setattr(target, "get_gemini_client", lambda: object())

    async def fake_gemini(client, model, contents, gen_config):
        captured["sys"] = gen_config.system_instruction
        return pytypes.SimpleNamespace(text="ok")

    monkeypatch.setattr(target, "_gemini_generate", fake_gemini)
    await target.query_target_conversation(
        [{"role": "user", "content": "hi"}], provider="gemini", model="m",
        system_prompt=None)
    assert captured["sys"] is None


def test_scenario_registry_has_secret_guardian():
    from scenarios import SCENARIOS, SECRET_GUARDIAN
    assert SCENARIOS["secret-guardian"] is SECRET_GUARDIAN
    assert SECRET_GUARDIAN.system_prompt == target.SYSTEM_INSTRUCTION
    assert SECRET_GUARDIAN.name == "secret-guardian"
