# Methods: Braun & Clarke’s Thematic Analysis Framework

This repository implements qualitative coding workflows guided by Braun & Clarke’s (2006) six-phase approach to thematic analysis.  
Each script corresponds to a specific phase or coding mode.

---

## Braun & Clarke’s Six Phases

1. **Familiarization with the Data**
   - Read and re-read the dataset
   - Begin taking notes
   - Identify broad ideas, repeated patterns, and initial impressions
   - *Script: `Step1_Familiarization_JZQA.py`*

2. **Generating Initial Codes**
   - Systematically code meaningful features of the dataset
   - Codes should be data-driven, not pre-defined
   - *Script: `Step2_InitialCoding.py`*

3. **Searching for Themes (Developing a Codebook)**
   - Group related codes into broader categories or themes
   - Begin creating definitions and examples
   - *Script: `Step3_CodebookDevelopment.py`*

4. **Reviewing Themes**
   - Refine themes, check against the dataset
   - Ensure coherence within themes and clear distinctions between them
   - (Optional; can be folded into Step 3 or extended manually)

5. **Defining and Naming Themes**
   - Finalize theme names and definitions
   - Clarify scope and focus of each theme
   - (Optional; often combined with Step 3 or later review)

6. **Producing the Report**
   - Write up results with supporting evidence from the dataset
   - Integrate thematic analysis into your research context
   - (Manual step outside of code)

---

## Deductive Coding

In addition to inductive analysis, this repository supports **deductive coding**:  
- Applying a pre-defined codebook to new transcripts  
- Useful for scaling codebook-driven analyses or validating themes  
- *Script: `Deductive_Coding.py`*

---

## Workflow Summary

1. **Inductive Phase**
   - Step 1: Familiarization → structured summary
   - Step 2: Initial Coding → list of codes from dataset
   - Step 3: Codebook Development → consolidated, defined, and exemplified codes

2. **Deductive Phase**
   - Apply the final codebook to new transcripts
   - Output includes coded datasets aligned with the established framework

---

## References

- Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77–101.
- Braun, V., & Clarke, V. (2019). Reflecting on reflexive thematic analysis. *Qualitative Research in Sport, Exercise and Health, 11*(4), 589–597.
