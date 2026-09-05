"""In-memory interface between OpenCOOD and an external transport."""

from collections.abc import Mapping


class CommunicationDataInterface:
    """Exchange module payloads without depending on a transport implementation."""

    def __init__(self) -> None:
        """Initialize empty outgoing and received payload stores."""
        self._outgoing: dict[str, dict[str, object]] = {}
        self._received: dict[str, dict[str, dict[str, object]]] = {}

    def publish(self, entity_id: str, module: str, payload: object) -> None:
        """Publish a module payload produced by an OpenCOOD entity.

        Parameters
        ----------
        entity_id : str
            Identifier of the sending CAV or RSU.
        module : str
            Name of the module that owns the payload contract.
        payload : object
            Transport-serializable module payload.
        """
        self._outgoing.setdefault(entity_id, {})[module] = payload

    def receive(self, receiver_id: str, sender_id: str, module: str) -> object | None:
        """Return a module payload delivered to an OpenCOOD entity.

        Parameters
        ----------
        receiver_id : str
            Identifier of the receiving ego entity.
        sender_id : str
            Identifier of the remote sending entity.
        module : str
            Name of the module that owns the payload contract.

        Returns
        -------
        object | None
            Delivered payload, or ``None`` when no matching payload exists.
        """
        return self._received.get(receiver_id, {}).get(sender_id, {}).get(module)

    def get_outgoing_payloads(self) -> Mapping[str, Mapping[str, object]]:
        """Expose payloads waiting for transport serialization.

        Returns
        -------
        collections.abc.Mapping[str, collections.abc.Mapping[str, object]]
            Outgoing payloads grouped by sender and module.
        """
        return self._outgoing

    def insert_received_payloads(self, receiver_id: str, sender_id: str, payloads: Mapping[str, object]) -> None:
        """Insert payloads delivered by an external transport.

        Parameters
        ----------
        receiver_id : str
            Identifier of the receiving ego entity.
        sender_id : str
            Identifier of the remote sending entity.
        payloads : collections.abc.Mapping[str, object]
            Module payloads delivered for the sender-receiver pair.
        """
        self._received.setdefault(receiver_id, {})[sender_id] = dict(payloads)

    def clear(self) -> None:
        """Discard all outgoing and received payloads."""
        self._outgoing.clear()
        self._received.clear()
