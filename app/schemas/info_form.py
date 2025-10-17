# schemas/info_form.py
"""
Info Form Schema.

Defines the `InfoForm` Pydantic model, which represents user-submitted
information for troubleshooting or escalation workflows. Supports validation,
text formatting for classification and RAG, Markdown rendering, and FastAPI
form integration.
"""

from fastapi import Form
from pydantic import BaseModel
from typing import Optional, Literal


class InfoForm(BaseModel):
    """
    Schema for collecting structured information about a ticket or call from the frontend.

    Attributes:
        workflow (Literal["troubleshooting", "escalation"]): Which workflow the form belongs to.
        input_type (Literal["ticket", "call"]): Type of input data being submitted.
        short_description (Optional[str]): Short description of the issue (tickets only).
        description (Optional[str]): Detailed description of the issue (tickets only).
        user_department (Optional[str]): Department of the user (tickets).
        extra_info (Optional[str]): Additional optional info (tickets).
        systems (Optional[str]): Systems or applications affected (calls only).
        technical_details (Optional[str]): Technical details provided (calls only).
        user_location (Optional[str]): User's department or location (calls only).
    """

    workflow: Literal["troubleshooting", "escalation"]
    input_type: Literal["ticket", "call"]

    short_description: Optional[str] = None
    description: Optional[str] = None
    user_department: Optional[str] = None
    extra_info: Optional[str] = None

    systems: Optional[str] = None
    technical_details: Optional[str] = None
    user_location: Optional[str] = None

    # ---------------------------
    # Validation
    # ---------------------------
    def validate(self) -> list[str]:
        """
        Validate the form fields depending on `input_type`.

        Returns:
            list[str]: A list of validation error messages. Empty if valid.
        """
        errors = []
        if self.input_type == "ticket":
            if not (self.short_description and self.short_description.strip()):
                errors.append("Short description is required for tickets.")
            if not (self.description and self.description.strip()):
                errors.append("Description is required for tickets.")
        elif self.input_type == "call":
            if not (self.systems and self.systems.strip()):
                errors.append("Systems/Applications field is required for calls.")
            if not (self.technical_details and self.technical_details.strip()):
                errors.append("Technical details are required for calls.")
        return errors

    # ---------------------------
    # Helpers
    # ---------------------------
    def to_classification_text(self) -> str:
        """
        Format the form into a classification-friendly text string.

        Returns:
            str: Concatenated short description and description (tickets),
            or systems and technical details (calls), separated by "####".
        """
        if self.input_type == "ticket":
            return f"{self.short_description} #### {self.description}"
        return f"{self.systems} #### {self.technical_details}"

    def to_full_info_text(self) -> str:
        """
        Format the form into a structured text representation.

        Returns:
            str: A multi-line string containing department, short description,
            description, and extra info (tickets), or systems, technical details,
            and user location (calls).
        """
        if self.input_type == "ticket":
            return (
                f"DEPT: {self.user_department}\n"
                f"SHORT: {self.short_description}\n"
                f"DESC: {self.description}\n"
                f"EXTRA: {self.extra_info}"
            )
        return (
            f"Systems/Apps: {self.systems}\n"
            f"Tech Details: {self.technical_details}\n"
            f"User Dept/Location: {self.user_location}"
        )
    
    def to_markdown(self) -> str:
        """
        Format form data into a Markdown bullet list, skipping empty fields.

        Returns:
            str: Markdown-formatted string summarizing the form.
        """
        parts = []

        if self.input_type == "ticket":
            if self.short_description:
                parts.append(f"- **SHORT DESCRIPTION:** {self.short_description}")
            if self.description:
                parts.append(f"- **DESCRIPTION:** {self.description}")
            if self.user_department:
                parts.append(f"- **DEPT:** {self.user_department}")

            if self.extra_info:
                parts.append(f"- **EXTRA INFO:** {self.extra_info}")
        else:  # call mode
            if self.systems:
                parts.append(f"- **Systems/Apps:** {self.systems}")
            if self.technical_details:
                parts.append(f"- **Tech Details:** {self.technical_details}")
            if self.user_location:
                parts.append(f"- **User Dept/Location:** {self.user_location}")

        return "\n".join(parts)


    # ---------------------------
    # Dependency integration
    # ---------------------------
    @classmethod
    def as_form(
        cls,
        workflow: str = Form(...),
        input_type: str = Form(...),
        short_description: str = Form(""),
        description: str = Form(""),
        user_department: str = Form(""),
        extra_info: str = Form(""),
        systems: str = Form(""),
        technical_details: str = Form(""),
        user_location: str = Form(""),
    ):
        return cls(
            workflow=workflow,
            input_type=input_type,
            short_description=short_description,
            description=description,
            user_department=user_department,
            extra_info=extra_info,
            systems=systems,
            technical_details=technical_details,
            user_location=user_location,
        )