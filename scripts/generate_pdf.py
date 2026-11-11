"""
Generate data/policy_brief.pdf programmatically using ReportLab.
Creates a clean, readable 1-page PDF document for PDF loader demonstrations.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_policy_pdf(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    lines = [
        ("Helvetica-Bold", 15, "Novasphere Dynamics - Autonomous Robotics Safety & Deployment Brief"),
        ("Helvetica", 10, "Document ID: SAF-2026-B | Access Level: Internal & Executive Compliance"),
        ("Helvetica", 10, ""),
        ("Helvetica-Bold", 12, "1. Purpose & Scope"),
        ("Helvetica", 10, "This safety brief defines mandatory field commissioning rules for autonomous mobile"),
        ("Helvetica", 10, "robots (AMRs) and automated warehouse drones connected to Novasphere OS."),
        ("Helvetica", 10, ""),
        ("Helvetica-Bold", 12, "2. Physical Safety Perimeter & Hardware E-Stop"),
        ("Helvetica", 10, "During active commissioning, personnel must maintain a minimum 3-meter safety"),
        ("Helvetica", 10, "perimeter around active AMR zones. Every robotic unit must carry a dual-channel"),
        ("Helvetica", 10, "hardware Emergency Stop (E-Stop) rated to SIL-2 / PLd compliance standards."),
        ("Helvetica", 10, ""),
        ("Helvetica-Bold", 12, "3. Wireless Kill-Switch & Telemetry Watchdog"),
        ("Helvetica", 10, "If telemetry heartbeat packets drop for more than 250 milliseconds in a shared"),
        ("Helvetica", 10, "human-robot aisle, the onboard safety controller triggers an automatic Category-1"),
        ("Helvetica", 10, "controlled deceleration stop within 1.2 meters of travel."),
        ("Helvetica", 10, ""),
        ("Helvetica-Bold", 12, "4. Executive Incident Escalation & Audit Retention"),
        ("Helvetica", 10, "Any near-miss safety event must be escalated to the Chief AI Ethics Officer"),
        ("Helvetica", 10, "(Dr. Elina Vance) and logged in the immutable compliance ledger within 4 hours."),
        ("Helvetica", 10, "Confidential quarterly safety audit reports are restricted to role='executive'."),
    ]

    y = height - 60
    for font_name, font_size, text in lines:
        c.setFont(font_name, font_size)
        if text:
            c.drawString(55, y, text)
        y -= 18 if text else 10

    c.showPage()
    c.save()
    print(f"Generated PDF successfully at: {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    generate_policy_pdf(root / "data" / "policy_brief.pdf")

