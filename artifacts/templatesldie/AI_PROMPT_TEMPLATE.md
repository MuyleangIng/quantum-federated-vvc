# Claude Code Prompt — Biweekly Meeting Template
# Paste this entire prompt into Claude Code every time you want to fill the template

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## FULL PROMPT — COPY EVERYTHING FROM HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I have a PowerPoint template file at this path:
/Users/ingmuyleang/A-Research/siminar_tmr/Biweekly Personal Meeting Template.pptx

Please do the following using python-pptx:

1. Read and inspect the template file to understand its exact layout, fonts,
   colors, placeholder positions, and slide structure.
2. Fill in the content I provide below — do NOT change any design element.
3. Handle figures and result images using the FIGURE RULES below.
4. Save the final file as: Biweekly_Meeting_[DATE].pptx
   in the same folder as the template.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMPLATE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Slide 1 — Title: "Bi-Weekly Meeting" + date + student name
Slide 2 — Index: research topic list
Slide 3 — Research section divider
Slide 4 — Executive Summary
Slide 5 — New Progress
Slide 6 — Response to Previous Feedback
Slide 7 — Discussion
Slide 8 — Next Action
Slide 9 — Q&A

NOTE: This template has RESEARCH SECTION ONLY — no project section.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — "New Progress" CONTINUATION SLIDES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If New Progress content is too long for one slide:
- Do NOT shrink the font or compress text
- Keep first slide title as: "New Progress"
- Add a new slide immediately after with title: "New Progress (cont)"
- Use the EXACT same slide layout as the original New Progress slide
- If still too long, add "New Progress (cont 2)", and so on
- Split content at natural bullet-point breaks

For items that are still running and not yet finished, write:
  "• [item name] — 🔄 Ongoing"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — FIGURES AND RESULT IMAGES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When a slide needs a figure, chart, or result image, follow this decision tree:

CASE A — I gave you a file path to an existing image:
  → Load that image file using add_picture() and place it on the slide.
  → Center it horizontally. Give it enough height to be readable.
  → Add a short caption below the image: "Figure X: [description]"

CASE B — I gave you data/numbers and asked for a chart:
  → Use matplotlib to generate the plot as a PNG file.
  → Save it as: /tmp/figure_[name].png
  → Load it into the slide using add_picture()
  → Add caption: "Figure X: [description]"
  → Example matplotlib code pattern:
      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(figsize=(7, 4))
      ax.plot(x_data, y_data, marker='o')
      ax.set_title('...'); ax.set_xlabel('...'); ax.set_ylabel('...')
      plt.tight_layout()
      plt.savefig('/tmp/figure_name.png', dpi=150, bbox_inches='tight')
      plt.close()

CASE C — No image file and no data provided (I just described what figure I want):
  → Do NOT leave the slide empty
  → Add a clearly visible placeholder text box on the slide with:

      ┌─────────────────────────────────────┐
      │  [FIGURE PLACEHOLDER]               │
      │  Figure X: [exact name I gave you]  │
      │  → Please add this image manually   │
      └─────────────────────────────────────┘

  → Style the box with a light gray background and dashed border if possible
  → Below it, write 2–3 bullet points explaining what this figure shows,
    so the audience understands the slide even before the image is added

CASE D — Mixed slide (some text + one figure):
  → Put text bullets on the LEFT half of the slide (width ~4.5 inches)
  → Put the figure on the RIGHT half (width ~4.5 inches)
  → If figure is placeholder (Case C), put placeholder box on right side

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — EMPTY OR MISSING SECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Never leave a slide with the original placeholder text like "(Edit accordingly...)".
If I did not provide content for a section, replace it with:
  - Discussion:             "No discussion items this meeting."
  - Response to Feedback:   "No feedback to address this period."
  - Next Action:            "Continuing current tasks. Details TBD."
  - Executive Summary:      "Work in progress — see New Progress slide."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MY CONTENT — FILL THIS IN BEFORE SENDING TO CLAUDE CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Date: [e.g. April 15, 2026]

Student Name: Ing Muyleang
Email: muyleanging@pukyong.ac.kr
University: Pukyong National University

════════════════════
RESEARCH SECTION
════════════════════

Research Name: [e.g. QAOA / VQE / etc.]

Executive Summary:
[2–4 sentences overview of this research period]

New Progress:
• [new result]
• [experiment result — 🔄 Ongoing]

Figures for New Progress slide:
• Figure 1: [name] — [file path OR "plot from data" OR "placeholder"]
  Data (if needed): [your numbers]

Response to Previous Feedback:
• [what you adjusted]

Discussion:
• [anything to discuss or get approval on]

Next Action:
• [Task 1]
• [Task 2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After building the PPTX:
1. Save as: Biweekly_Meeting_[DATE].pptx  (same folder as template)
2. Print a summary of every slide: slide number, title, content added,
   and whether any figure placeholders were inserted that I need to fill manually.
3. List all [FIGURE PLACEHOLDER] slides clearly at the end so I know
   exactly which images I still need to add myself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## END OF PROMPT — STOP COPYING HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
