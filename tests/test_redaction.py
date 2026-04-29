from agentlock import RedactionRule, Redactor


def test_redactor_masks_nested_paths() -> None:
    redactor = Redactor(redact=["customer.email", "customer.phone"], mode="mask")

    redacted = redactor.redact(
        {
            "customer": {
                "email": "casey@example.com",
                "phone": "+1-555-0100",
                "name": "Casey",
            }
        }
    )

    assert redacted["customer"]["email"] == "***REDACTED***"
    assert redacted["customer"]["phone"] == "***REDACTED***"
    assert redacted["customer"]["name"] == "Casey"


def test_redactor_supports_remove_and_hash_modes() -> None:
    redactor = Redactor(
        redact=[
            RedactionRule(path="customer.email", mode="remove"),
            RedactionRule(path="customer.phone", mode="hash"),
        ]
    )

    redacted = redactor.redact(
        {
            "customer": {
                "email": "casey@example.com",
                "phone": "+1-555-0100",
            }
        }
    )

    assert "email" not in redacted["customer"]
    assert redacted["customer"]["phone"].startswith("sha256:")
