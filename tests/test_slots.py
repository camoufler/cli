"""Tests for speaker polarity and stated-tone helpers."""

from camoufler.slots import (
    USER_VOICE_ROLE,
    helper_writing_role,
    is_explicit_role_request,
    is_user_voice_writing,
    role_inverts_speaker,
    stated_tone_clause,
)

BLENDER = (
    "I wanna write an email about my ninja blender, it stop working and I m pissed of."
)
ANGRY_BOSS = "Can you help me draft an angry email to my boss?"
ACT_AS_SUPPORT = "Act as Ninja customer support and reply to this complaint."
HEADLINES = "Draft three short headlines for an open-source analytics dashboard."


def test_blender_is_user_voice_not_explicit_role():
    assert is_user_voice_writing(BLENDER) is True
    assert is_explicit_role_request(BLENDER) is False
    assert helper_writing_role(BLENDER) == USER_VOICE_ROLE
    assert role_inverts_speaker("customer support representative", BLENDER) is True


def test_angry_email_is_user_voice():
    assert is_user_voice_writing(ANGRY_BOSS) is True
    assert helper_writing_role(ANGRY_BOSS) == USER_VOICE_ROLE
    assert role_inverts_speaker("the boss", ANGRY_BOSS) is True


def test_act_as_keeps_assigned_role():
    assert is_explicit_role_request(ACT_AS_SUPPORT) is True
    assert helper_writing_role(ACT_AS_SUPPORT) is None
    assert role_inverts_speaker("customer support representative", ACT_AS_SUPPORT) is False


def test_marketing_copy_is_not_user_voice():
    assert is_user_voice_writing(HEADLINES) is False
    assert helper_writing_role(HEADLINES) is None
    assert role_inverts_speaker("direct-response B2B copywriter", HEADLINES) is False


def test_stated_tone_from_affect():
    assert stated_tone_clause(BLENDER) == "Use a frustrated, direct, and firm tone"
    assert stated_tone_clause(ANGRY_BOSS) == "Use an angry, direct, and firm tone"
    assert stated_tone_clause(HEADLINES) is None
    mixi = "Write an email about my mixi not working. Express how disappointed I am."
    assert stated_tone_clause(mixi) == "Use a disappointed tone"
