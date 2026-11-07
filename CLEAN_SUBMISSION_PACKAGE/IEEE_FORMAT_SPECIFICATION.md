# IEEE Conference & Transactions Formatting Specification Rules

> **Standard:** Official IEEE Conference / Transactions Manuscript Template  
> **Source:** IEEE Style Manual & Author Center Standards  
> **Applicability:** All manuscripts, LaTeX documents, Word documents, and text components generated for IEEE submissions.

---

## 1. Title & Metadata Rules
1. **Paper Title:**
   - Centered, bold, Title Case (e.g., 24 pt in Word / `\title` in IEEEtran).
   - **CRITICAL RESTRICTION:** Absolutely **NO** symbols, special characters, footnotes, math equations, or acronyms in the paper title.
   - **NO Sub-titles:** Sub-titles are not indexed in IEEE Xplore and must **NOT** be used.
2. **Authors & Affiliations:**
   - Minimum 1 author; standard template handles up to 6 authors in horizontal row layout (left to right, then next row).
   - Author block lines:
     - Line 1: Given Name Surname
     - Line 2: Department name of organization (of Affiliation)
     - Line 3: Name of organization (of Affiliation)
     - Line 4: City, Country
     - Line 5: Email address or ORCID
   - Do **NOT** list names in vertical single columns; do **NOT** group by affiliation. Keep affiliations succinct.
3. **Abstract:**
   - Format: Run-in style `Abstract—` (bold/italic with em-dash).
   - **CRITICAL RESTRICTION:** Absolutely **NO** symbols, special characters, footnotes, citations, or mathematical formulas in the Abstract. Plain text only.
4. **Keywords:**
   - Format: Run-in style `Keywords—` followed by 4–6 comma-separated terms (capitalized, no period at end).

---

## 2. Heading Hierarchy & Numbering
1. **Heading 1 (Primary Text Head):**
   - Format: Uppercase Roman Numerals (e.g., `I. INTRODUCTION`, `II. BACKGROUND AND RELATED WORK`, `III. MODEL ARCHITECTURE`).
   - Centered or small-caps bold.
2. **Heading 2 (Sub-Head):**
   - Format: Uppercase Letters (e.g., `A. Selective State Spaces`, `B. Decayed Associative Neural Memory`).
   - Left-aligned, italicized.
3. **Heading 3 (Sub-Sub-Head):**
   - Format: Numbers with parentheses and run-in text (e.g., `1) Zero-Order Hold Discretization:`).
   - Indented, italicized, run-in with paragraph.
4. **Component Heads (Unnumbered):**
   - `ACKNOWLEDGMENT` and `REFERENCES` are component heads, not topically subordinate. Never number them (do **NOT** write `XI. REFERENCES`).

---

## 3. Equations & Mathematical Typography
1. **Equation Placement & Numbering:**
   - Centered horizontally using center tab stop.
   - Equation numbers in parentheses, positioned **flush right**, as in `(1)`.
   - Number equations consecutively.
2. **Equation Citation in Prose:**
   - Refer strictly to `(1)`, **NOT** `Eq. (1)` or `equation (1)`, except at the very beginning of a sentence: `Equation (1) is...`
3. **Variable Styling:**
   - Italicize Roman symbols for quantities and variables ($x$, $y$, $S$, $h_t$).
   - Do **NOT** italicize Greek symbols ($\Delta$, $\alpha$, $\beta$, $\mu$).
   - Use an en-dash / minus sign ($-$), never a hyphen, for minus signs and negative numbers.
4. **Punctuation:**
   - Equations are part of grammatically complete sentences and must be punctuated with ending periods or commas.
   - Every variable must be defined either immediately preceding or immediately following the equation.

---

## 4. Figures and Tables
1. **Placement:**
   - Position figures and tables strictly at the **top** or **bottom** of columns. Avoid placing them in the middle of a column.
   - Large figures and tables may span across both columns (`figure*` / `table*`).
2. **Figures:**
   - Captions placed **below** the graphic: `Fig. 1. Figure caption text.`
   - In text, always use the abbreviation `Fig. 1`, even at the beginning of a sentence.
   - Axis labels: Use words and units in parentheses, e.g., `Magnetization (A/m)`, never isolated abbreviations like `M`.
   - Figure labels and annotations: 8 pt font.
3. **Tables:**
   - Table heads appear **above** the table:
     `TABLE I` (centered, Roman numerals)  
     `PERFORMANCE COMPARISON ACROSS ARCHITECTURES` (small caps/uppercase, centered).
   - Use formal borders: top rule, mid rule under headers, bottom rule.
   - Footnotes in tables use lowercase letters ($^a$, $^b$).

---

## 5. Units & Numerals
- Primary units must be SI (MKS) or CGS.
- Never mix units (e.g., amperes and oersteds in the same equation).
- Do not mix unit spellings and abbreviations (use `Wb/m²` or `webers per square meter`, never `webers/m²`).
- Always write a leading zero before decimal points: `0.25`, never `.25`.
- Write `cm³`, not `cc`.

---

## 6. Style, Grammar & Lexical Rules
1. **"Data":** The word *data* is plural, not singular (e.g., "the data were collected", not "the data was collected").
2. **Physical Constants:** Subscript for vacuum permeability is $\mu_0$ (zero subscript), never lowercase letter $o$.
3. **Punctuation & Quotes:**
   - Commas and periods go inside quotation marks only when citing complete thoughts or titles.
   - When quotation marks highlight a single term, punctuation goes outside.
4. **Terminology:**
   - Avoid "essentially" to mean approximately or effectively.
   - Use "alternatively" instead of "alternately".
   - Distinguish homophones: *affect* (verb) vs. *effect* (noun), *complement* vs. *compliment*, *discrete* (separate) vs. *discreet*.
   - The prefix *non* is joined without a hyphen (e.g., *nonlinear*, *nonoscillating*).
   - "et al." has a period after *al.*, but **no** period after *et*.
   - Abbreviations: "i.e.," and "e.g.," must have trailing periods and commas.

---

## 7. Acknowledgment
- Spelling is strictly American: **Acknowledgment** (no 'e' after 'g').
- Sponsor and funding notices go in an unnumbered footnote on the first page.
- Do not write "one of us (A. B.) thanks...", write "A. B. thanks...".

---

## 8. References Formatting
1. **In-Text Citation:**
   - Number citations consecutively in square brackets: `[1]`, `[2]`.
   - Punctuation follows the bracket: `...as demonstrated in [1].`
   - Refer simply to `[1]`, **NOT** `Ref. [1]` or `reference [1]`, except at the start of a sentence: `Reference [1] was the first...`
2. **Reference List:**
   - Give all authors' names unless there are six or more; do **NOT** use "et al." unless $\ge 6$ authors.
   - Capitalize only the first word in a paper title, except for proper nouns and element symbols.
   - Format:
     - **Journal:** J. K. Author, "Name of paper," *Abbrev. Title of Periodical*, vol. x, no. x, pp. xxx–xxx, Month, Year.
     - **Conference:** J. K. Author, "Title of paper," in *Proc. Name of Conf.*, City, Country, Year, pp. xxx–xxx.
     - **Book:** J. K. Author, *Title of Book*, xth ed. City: Publisher, Year, pp. xxx–xxx.
     - **Preprint / Online:** J. K. Author, "Title of paper," Year, arXiv:xxxx.xxxxx. [Online]. Available: URL
