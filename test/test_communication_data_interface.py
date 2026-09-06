"""Tests for the transport-neutral OpenCOOD communication interface."""

from opencood.communication import CommunicationDataInterface


def test_publishes_and_replaces_outgoing_module_payloads() -> None:
    interface = CommunicationDataInterface()

    interface.publish("cav-1", "detector", {"value": 1})
    interface.publish("cav-1", "detector", {"value": 2})

    assert interface.get_outgoing_payloads() == {"cav-1": {"detector": {"value": 2}}}


def test_inserts_and_receives_transport_payloads() -> None:
    interface = CommunicationDataInterface()
    payload = {"detections": [1, 2, 3]}

    interface.insert_received_payloads("cav-1", "cav-2", {"detector": payload})

    assert interface.receive("cav-1", "cav-2", "detector") is payload
    assert interface.receive("cav-1", "missing", "detector") is None


def test_clear_discards_both_payload_directions() -> None:
    interface = CommunicationDataInterface()
    interface.publish("cav-1", "detector", {"outgoing": True})
    interface.insert_received_payloads("cav-1", "cav-2", {"detector": {"received": True}})

    interface.clear()

    assert interface.get_outgoing_payloads() == {}
    assert interface.receive("cav-1", "cav-2", "detector") is None
