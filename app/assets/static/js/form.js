// assets/static/form.js

document.addEventListener("DOMContentLoaded", () => {
  const inputRadios = document.querySelectorAll('input[name="input_type"]');
  const ticketForm = document.getElementById("ticket-form");
  const callForm = document.getElementById("call-form");

  function updateFormVisibility() {
    const selected = document.querySelector('input[name="input_type"]:checked');
    if (!selected) return;

    const value = selected.value;
    if (ticketForm) ticketForm.classList.toggle("hidden", value !== "ticket");
    if (callForm) callForm.classList.toggle("hidden", value !== "call");
  }

  inputRadios.forEach(radio => {
    radio.addEventListener("change", updateFormVisibility);
  });

  // Initial run (on page load)
  updateFormVisibility();
});

function updateButton(mode) {
  const btn = document.getElementById("submit-btn");

  if (mode === "escalation") {
    btn.textContent = "Suggest Group";
  } else if (mode === "troubleshooting") {
    btn.textContent = "Troubleshoot";
  } else {
    btn.textContent = "Classify";
  }
}

// attach listeners to both radios
document.querySelectorAll('input[name="workflow"]').forEach(el => {
  el.addEventListener("change", e => updateButton(e.target.value));
});

// run once on load to sync button with whichever is checked
const checked = document.querySelector('input[name="workflow"]:checked');
if (checked) updateButton(checked.value);