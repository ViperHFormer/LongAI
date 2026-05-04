from longai.schema.models import EvidencePack, SpeakerRole


def test_evidence_pack_model():
    pack = EvidencePack(
        segment_id="s1",
        session_id="session_0000",
        start_time=0.0,
        end_time=1.0,
        waveform_path="a.wav",
        speaker_role=SpeakerRole.WEARER,
    )
    assert pack.segment_id == "s1"
