# GLOBALNAV demo paper revision report

Date: 2026-07-08

## What was changed in this pass

1. Reframed the paper for a demo submission.
   - The title, abstract, introduction, contributions, conclusion, and limitations now focus on the GLOBALNAV route-planning demo rather than instruction following.
   - The main experiments now report only system-level success on GlobNav-Bench and the external OD parser/clarification stress test.
   - Instruction-follower evaluation is moved to the appendix as an auxiliary use of the benchmark annotations.

2. Preserved the benchmark's instruction-following relevance without making it a main demo claim.
   - GlobNav-Bench is still described as containing optional route-playback/procedural annotations.
   - The text now says these annotations support future instruction-following work, while the demo evaluation focuses on request resolution and feasible route construction.

3. Removed visible TODO placeholders.
   - The prompt appendix now gives concise implementation-level descriptions and code paths instead of red TODO markers.

4. Improved citation coverage and fixed citation metadata.
   - Added citations for Nominatim, OpenFlights, and nDTW.
   - Added fine-grained citations for classical multimodal route planning and language-agent travel planning.
   - Added Introduction citations for LLM scaling, chain-of-thought reasoning, and LLM spatial reasoning/path-generation capability.
   - Renamed the Overpass citation key from `wiki:xxx` to `openstreetmap_wiki_overpass_2026`.
   - Corrected the MultiWOZ 2.4 BibTeX authors according to ACL Anthology. The previous local entry incorrectly included Qiao Zhang and Shenghui Li.
   - Aligned the ATIS entry with ACL Anthology's exported BibTeX.
   - Updated OSRM and Schema-Guided Dialogue metadata using DOI/Crossref records, and protected proper-name capitalization in BibTeX titles.

## Citation audit

I checked the existing and newly added references against primary or official sources where possible.

| Citation key | Status | Source used |
|---|---|---|
| `Anderson_2018_CVPR` | OK | CVF official BibTeX for R2R / VLN |
| `Chen_2019_CVPR` | OK | CVF official BibTeX for Touchdown |
| `Schumann_2024` | OK | AAAI / DOI metadata for VELMA |
| `Xu_2025` | OK | AAAI / DOI metadata for FLAME |
| `NEURIPS2020_1457c0d6` | Added | NeurIPS proceedings BibTeX for GPT-3 / few-shot LMs |
| `NEURIPS2022_9d560961` | Added | NeurIPS proceedings BibTeX for chain-of-thought prompting |
| `yao2023react` | OK | OpenReview / ICLR metadata; existing entry retained |
| `NEURIPS2023_d842425e` | OK | NeurIPS proceedings BibTeX for Toolformer |
| `pmlr-v235-xie24j` | Added | PMLR official BibTeX for TravelPlanner |
| `rizvi-etal-2024-sparc` | Added | ACL Anthology official BibTeX for LLM spatial reasoning |
| `bast-etal-2016-route-planning` | Added | Crossref/DOI metadata for the Springer Algorithm Engineering chapter |
| `Haklay_2008` | OK | Crossref/DOI metadata for the OSM paper |
| `luxen-vetter-2011` | Updated | Crossref/ACM DOI metadata for the ACM SIGSPATIAL paper |
| `openstreetmap_wiki_overpass_2026` | OK as web citation | OpenStreetMap Wiki cite-page metadata; Overpass has no standard archival paper in the current paper context |
| `nominatim_2026` | Updated web citation | Official Nominatim manual; no archival paper citation was found |
| `openflights_data_2026` | OK as web citation | Official OpenFlights data page; no archival paper citation was found for the data source itself |
| `ilharco-etal-2019-ndtw` | OK for appendix use | Google Research publication page / Google Scholar metadata; workshop publication |
| `ye-etal-2022-multiwoz` | Fixed | ACL Anthology official BibTeX |
| `rastogi-etal-2020-schema` | Updated | AAAI DOI/Crossref metadata |
| `hemphill-etal-1990-atis` | Fixed | ACL Anthology official BibTeX |

Notes:

- For formally published papers, I used official proceedings, DOI, ACL Anthology, CVF, NeurIPS, OpenReview, or ACM-style metadata.
- For tools/data sources without formal publication citations in the paper context, I used official project/manual/data pages as `@misc` entries rather than inventing publication metadata.
- The nDTW citation is now only used in the appendix follower experiment, where it is directly relevant.
- The compiled paper now has no missing citation keys and no unused BibTeX entries.

## Remaining improvement suggestions

These are not blockers for the current structural rewrite, but I would consider them before submission.

1. Replace the schema placeholder figure.
   - Current source still contains `\placeholder{\bench annotation schema}`.
   - For a demo paper, a real schema figure or compact table would look much more credible.
   - Priority: high.

2. Strengthen reproducibility for Table 2, the system-level success table.
   - The paper reports system-level numbers for several LLM configurations, but the current text does not point to a run manifest, command, or per-example output for that table.
   - If those runs exist, add a short appendix note or artifact path. If they are provisional, replace them with verified runs.
   - Priority: high.

3. Confirm exact model names and dates.
   - Names such as `GPT-5.5`, `Claude Sonnet 4.6`, and `Gemini 2.5 Pro` should match the exact public/API model names used at evaluation time.
   - A short note like "evaluated on DATE with API model ID ..." would prevent reviewer confusion.
   - Priority: high.

4. Clarify route validity more concretely.
   - The current definition says route validity requires support from annotated evidence. This is reasonable, but a reviewer may ask what counts as evidence for OSRM, Overpass transit, OpenFlights, ferries, or fallback estimates.
   - A short appendix table mapping each route type to evidence policy would help.
   - Priority: medium-high.

5. Consider moving some external OD details to appendix if the demo format is short.
   - The external OD evaluation is useful because it supports the system-success claim, but the protocol paragraph is dense.
   - If page pressure becomes serious, keep Table 3 and one summary paragraph in the main text, and move derivation details to appendix.
   - Priority: medium.

6. Improve the demo figure/case-study presentation.
   - The GUI screenshot caption is now stronger, but the paper would benefit from one concrete walk-through: user request -> clarification if any -> route segments -> user option switch.
   - This is especially valuable for a demo submission.
   - Priority: medium.

7. Decide whether the follower appendix should remain.
   - Keeping it is useful because the benchmark genuinely contains those annotations.
   - If the venue has strict demo length constraints, the appendix can be shortened to a one-paragraph "secondary benchmark use" statement.
   - Priority: medium.

8. Add a short artifact/release statement.
   - A demo paper is stronger if it clearly says what will be released: code, benchmark samples, evaluation scripts, cached metadata, GUI instructions, and limitations on live API keys/data access.
   - Priority: medium.
