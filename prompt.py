SELECTION_EXTRACTION_PROMPT = r"""
You are extracting HISTORICAL COLLEGE PLACEMENT SELECTION DATA from a TPO email.

The ONLY target is an email that reports an actual placement/recruitment selection outcome.
Examples:
- "Congratulations, the following students have been selected..."
- "Selections - Company..."
- "More Selections..."
- "Students selected through campus placement drive..."

Do NOT treat these as selection emails:
- registration notices
- job/opportunity announcements with no selection outcome
- test invitations
- interview schedules
- general information
- unrelated college emails

Return ONLY valid JSON matching the supplied schema.

Rules:
1. is_selection_email must be false when the email does not report an actual selection.
2. If false, records must be [].
3. One email may contain multiple selection records when there are clearly different companies, roles, batches, or selection groups.
4. Never invent missing values. Use null for missing scalar values and [] for missing lists.
5. Preserve facts from the email; do not infer a package or eligibility criterion.
6. A selected-student table is strong evidence of selection.
7. If the email contains a selected-student table, extract each student when names are present.
8. selected_count should be the stated selected count when explicitly given. If a complete student table is present and no count is stated, it may be counted.
9. Keep CTC, base salary and stipend separate.
10. selection_date is the date of the selection/result only when supported by the email.
11. source_subject and source_message_id should be copied from the source metadata.
12. Do not include raw email prose in fields.
13. Normalize obvious whitespace, but do not change the meaning.
"""
