import uuid

def generate_unique_id(reference: str = None, length: int = None) -> str:
    """
    Generate a unique identifier as a hexadecimal string.

    The identifier can be:
        - Reproducible based on input text (UUID5) if `reference` is provided.
        - Completely random (UUID4) if no `reference` is given.

    Args:
        reference (str, optional): Text to generate a reproducible UUID. Defaults to None.
        length (int, optional): Maximum length of the returned string (up to 32). Defaults to None.

    Returns:
        str: Hexadecimal string representing the UUID, truncated to `length` if specified.
    """
    if reference is not None:
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, reference).hex
    else:
        unique_id = uuid.uuid4().hex

    if length is not None and length > 0:
        length = min(length, 32)
        return unique_id[:length]

    return unique_id
