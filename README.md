# vacancy-match

Collect vacancies from public job APIs, rank them against your resumes, then apply your own terms — deal-breakers,
wishes, salary floor, red lines — as a second, explained stage. Learn from every fit / no-fit decision, see what
happens after you apply, and measure all of it on real decisions instead of eyeballing.

```
## 4. Python Developer (client-facing) — CallStack
- score 30 · fit 60 · track A-backend · $2600 per month
- Why: Projects: Data pipeline (0.82); Summary (0.81)
- Adjusted: language ×0.5
- Similar labeled: fit 0.94 Python Backend Developer; fit 0.92 Telegram Bot Developer; no_fit 0.92 Call Center Agent; ...
```

The vacancy matches the resume (fit 60), but it needs fluent German, which this candidate does not speak. The score
halves and the job drops below the ones they would actually take. Every number on the card says where it came from.

## How the score works

Every vacancy gets two numbers: **fit** — does it look like my experience? — and **score** — would I take it?

### Stage 1: fit, resume against vacancy

1. **Chunks.** Each resume (Markdown, one per direction you apply for — a *track*) is split into chunks: one per
   section and one per project. A whole resume as one vector blurs everything into an average; a vacancy about data
   pipelines should meet the pipeline project.
2. **Embeddings.** Vacancies and chunks become vectors with a local multilingual model
   (`intfloat/multilingual-e5-base` by default — no API key, CPU is enough). The model reads about 512 tokens, so the
   embedded text puts requirements and conditions first and the company story last.
3. **Best match.** A vacancy's similarity is its closest chunk within a track; the best track wins and is shown with
   the two chunks that matched.
4. **0–100.** Raw similarity lives in a band a few hundredths wide, where "a little lower" and "much lower" look the
   same. So it is mapped to 0–100 against all known vacancies: the median vacancy gets 50, the 10th and 90th
   percentiles land near 10 and 90. The scale is frozen per model, so a vacancy keeps its fit as the base grows;
   `vm calibrate` re-measures it on purpose.

### Stage 2: score, the candidate's terms

**score = fit × the effect of every term the vacancy hits.** Without a rules file, score = fit. With the example
candidate below (target $1500, floor $1100):

| vacancy | fit | effects | score |
|---|---|---|---|
| Python developer, fluent German required, $1300 | 80 | language ×0.5 · salary $1300 ×0.817 | 33 |
| Data engineer, open-source tools, $1600 | 60 | open_source ×1.1 | 66 |
| Near-perfect match, on-call rotation, $1250 | 95 | on_call ×0.9 · salary $1250 ×0.733 | 63 |
| Data entry, "pay a registration fee for your starter kit" | 40 | upfront_fee: cut | bottom of the list, reason shown |

Effects multiply instead of subtracting, so a near-perfect vacancy with one serious miss can still beat a mediocre
one with none. A cut sends a vacancy below every other one whatever its fit.

### Feedback from your decisions

Labeled vacancies vote: a new vacancy close to three rejected scams moves down, one close to jobs you took moves up —
by up to 15 points of fit. The vote switches on once there are at least 100 labels, 20 of them rejections; until
then it would mostly echo a handful of decisions.

## Your terms: wishes, conditions, red lines

Similarity answers "does this look like my experience?". It does not answer "would I take it?". A job that needs
daily calls in a language you do not speak, pays half your floor, or is a scam can look like a perfect match on paper
and land right next to the jobs you want. Real examples of terms people set: *no live calls in English — I write
well, speak poorly* (a condition); *never receive client payments on my own card* (a red line).

Terms are regular expressions in a YAML file, grouped by how much they matter:

| tier | for | default effect | example (fictional candidate) |
|---|---|---|---|
| flag | worth seeing, not a reason | shown only | `agency`: outstaff, staffing agency |
| wish | nice to have | × 0.9 | `on_call`: on-call rotation, 24/7 support |
| condition | a deal-breaker that a regex can still misread | × 0.5 | `language`: fluent German · `onsite`: office-based |
| boost | what you want most | × 1.1 | `open_source`: open-source work |
| red line | never worth reading | cut | `upfront_fee`: "pay for training / a starter kit" |
| salary | target and floor | linear between points, cut below the floor | $1500 target, $1100 floor |
| stale | old postings | × 0.8 | older than 30 days |

```yaml
wishes:
  on_call: ["on-?call", "24/7 support"]
conditions:
  language: ["fluent german", "german \\(?c1", "native german"]
red_lines:
  upfront_fee: ["pay (for|a) (training|starter kit|registration)", "registration fee"]
salary:   # monthly USD: $1500 and up no effect, $1350 ×0.9, $1200 ×0.65, $1100 ×0.5, below $1100 cut
  curve: {1500: 1.0, 1350: 0.9, 1200: 0.65, 1100: 0.5}
  rates: {USD: 1.0, EUR: 1.08, GBP: 1.27}
```

The whole file: [`examples/rules.example.yaml`](examples/rules.example.yaml). `levels:` changes the tier defaults;
any rule can set its own, e.g. `office: {patterns: ["office"], factor: 0.7}`.

- **Why a condition halves instead of cutting:** a regex misreads ("German is a plus"), so a serious miss costs half
  and a near-perfect vacancy still surfaces with the reason next to it. Cuts are for what is never worth reading.
- **Salary:** a range counts at its middle (1000–1400 → 1200). Part-time is read from the title, the work format
  and the salary line — a body says "this is not part-time" as often as "part-time". Its pay is scaled to full time
  when the hours are stated anywhere (20 h/week at $900 ≈ $1660 full time); without hours it is only flagged. Hourly
  pay counts 160 hours a month. A figure with no period that would be over $15,000 a month is read as a yearly salary:
  US and EU boards rarely write "per year". Pay per project is flagged, not judged against a monthly floor. An
  unknown salary, or a currency missing from `rates:`, is flagged (`no_salary`), never penalised: ask.
- **Tuning:** the factors are starting guesses. `vm eval` reports how often each rule fired on vacancies you took and
  on ones you rejected; a penalty that hits as many fits as no-fits is too strict.

## Quick start

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). The default model (~1 GB) downloads on first use; no API key.

```bash
git clone https://github.com/Augustforge/vacancy-match && cd vacancy-match
uv sync --extra model
uv run python examples/make_synthetic.py
uv run vm --db demo.db index-profile examples/profile
uv run vm --db demo.db add examples/vacancies.jsonl
uv run vm --db demo.db label --file examples/labels.jsonl
uv run vm --db demo.db rank --run demo-2026-10-02 --rules examples/rules.example.yaml
uv run vm --db demo.db eval --labels all --models intfloat/multilingual-e5-base,hash-512 --rules examples/rules.example.yaml
uv run vm funnel examples/tracker.example.csv --today 2026-10-15 --wait-channel board=7
```

## Use it on your own search

1. **Resumes.** One Markdown file per direction you apply for, named `resume-<TRACK>-<LANG>.md`
   (`resume-backend-EN.md`, `resume-data-RU.md`). `##` sections and `**bold**` project lines become chunks.
2. **Your terms (optional).** Copy `examples/rules.example.yaml` and write your own salary curve, conditions, wishes
   and red lines. Skip it and the ranking is resume fit alone.
3. **Vacancies.** `vm fetch` collects them from the boards' own public APIs — no scraping, no login — into JSONL,
   one run per file (see the table below). Any other source works too: JSONL with `source`, `url`, `title` and
   ideally `company`, `text`, `salary_text`, `published_at`, `work_format` (full-time, part-time, project) and
   `location`. The run id must contain a date (`2026-10-14`, `links-2026-10-14`): runs are replayed in date order.
4. **Run it:**

```bash
export VM_DB=my.db VM_RULES=my-rules.yaml          # or pass --db / --rules
uv run vm index-profile path/to/resumes
uv run vm fetch remotive:python remoteok:python greenhouse:gitlab --where remote --out run.jsonl
uv run vm add run.jsonl --run 2026-10-14
uv run vm calibrate                                # once you have 50+ vacancies; again after a model or resume change
uv run vm rank --run 2026-10-14 --out top.md
uv run vm label <url> fit --actor human --track backend          # every decision, fit and no_fit
uv run vm label <url> no_fit --actor human --reasons onsite,low_pay   # reasons are your own words
uv run vm eval --rules my-rules.yaml               # after a few runs with real rejections
```

### Where `vm fetch` collects from

| source | what | argument |
|---|---|---|
| `greenhouse:<board>` | one company's open jobs on Greenhouse | the name in `job-boards.greenhouse.io/<board>` |
| `lever:<company>` | one company's open jobs on Lever | the name in `jobs.lever.co/<company>` |
| `ashby:<org>` | one company's open jobs on Ashby | the name in `jobs.ashbyhq.com/<org>` |
| `remotive:<search>` | remote jobs, Remotive | a search, e.g. `python` |
| `remoteok[:<tag>]` | remote jobs, Remote OK | optional tag, e.g. `python` |
| `himalayas:<search>` | remote jobs, Himalayas | a search |
| `arbeitnow` | jobs in Germany and remote, Arbeitnow | — |
| `hh:<search>` | HeadHunter (hh.ru, hh.kz, …) | needs `HH_TOKEN`: since 2026 the hh API answers only registered apps ([dev.hh.ru](https://dev.hh.ru)) |
| `page:<url>` | one vacancy page you found yourself | any page with schema.org `JobPosting` markup — hh vacancy pages have it, no token needed |

`--where REGEX` keeps jobs whose title, location or work format match (`--where remote`). A source that fails
(a typo in a board name, no network) is reported and the others are still collected. Every record keeps the job's
own URL and its source: the aggregators ask you to link back and credit them if you republish their jobs, and
Remotive asks for no more than a few requests a day.

## After you apply: the funnel

Ranking decides what to send; the funnel shows what happened to it. Keep a CSV tracker
([`examples/tracker.example.csv`](examples/tracker.example.csv)):

| column | meaning |
|---|---|
| `company` | required |
| `sent` | `YYYY-MM-DD`, required unless the row is a draft |
| `status` | `draft` (not confirmed sent) · `sent` · `replied` · `interview` · `offer` · `rejected` · `closed` (you stopped) |
| `channel` | how you sent it: email, form, board, a messenger |
| `track`, `source` | which resume; where you found the vacancy |
| `replied` | date of their first answer — keep it when you close a thread that did get one |
| `updated` | the last message either way, e.g. your reminder: it restarts the silence clock |
| `note` | free text |

```bash
uv run vm funnel tracker.csv --wait 5 --wait-channel board=7 --out funnel.md
```

The report lists whom to remind (no answer for 5 working days since the last message, or the channel's own wait) —
one reminder each: a thread still silent after it moves to "let it go". Then open threads, interviews included,
where they answered and the next step is pending; applications per week and how many got an answer; the reply rate
by channel, resume and source; drafts that still need a "sent or not"; and rows it could not read. From the example
tracker:

```
Sent 8 · replied 3 (38%) · interviews 1 · offers 0. Not confirmed sent: 1.

## Follow up
| sent | company | channel | track | working days silent |
| 2026-10-01 | Metricly | board | A-backend | 10 |
| 2026-10-06 | Chatly | form | A-backend | 7 |
| 2026-10-08 | Lumen Data | email | A-backend | 5 |

## Reply rate by track
| track | sent | replied | rate | interviews |
| A-backend | 5 | 1 | 20% | 1 |
| B-design | 3 | 2 | 67% | 0 |
```

## Evaluation

`vm eval` replays runs in date order — neighbours come only from earlier runs, so labels never leak — and reports,
for several models and settings at once, each with and without stage 2:

- **AUC** (the main metric): the share of fit/no-fit pairs in the right order, ties count half; 0.5 is random.
- **AP** rewards fits near the top. **np@k** (hits / min(k, fits)) and **recall@k** are shown only for runs larger
  than k: inside k every ordering scores the same. Raw precision@k on small runs mostly measures the share of fits.
- **Pooled AUC with a bootstrap 95% interval** — read the interval before the number.
- **Track accuracy**, the worst misses both ways, and the per-rule report.

It counts your own decisions by default: an agent's labels would only measure the system agreeing with itself
(`--labels all` includes them).

Synthetic demo — it shows the mechanics, it is not a benchmark. 14 vacancies; two of them share the candidate's
skills but not their terms (fluent German; $600 a month), so only stage 2 can send them down:

| config | np@10 | np@20 | recall@20 | AUC | AP | track accuracy | pooled AUC [95% CI] |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base nw=0.0 hybrid=0.0 stage=1 | — | — | — | 0.96 | 0.95 | 1.00 | 0.98 [0.88–1.00] |
| intfloat/multilingual-e5-base nw=0.0 hybrid=0.0 stage=1+2 | — | — | — | 1.00 | 1.00 | 1.00 | 1.00 [1.00–1.00] |
| hash-512 nw=0.0 hybrid=0.0 stage=1 | — | — | — | 0.85 | 0.83 | 1.00 | 0.92 [0.75–1.00] |
| hash-512 nw=0.0 hybrid=0.0 stage=1+2 | — | — | — | 1.00 | 1.00 | 1.00 | 1.00 [1.00–1.00] |

The rule report on the same labels:

| rule | effect | on fit | on no_fit | share of fits | share of no-fits |
|---|---|---|---|---|---|
| language | ×0.5 | 0 | 2 | 0% | 25% |
| no_salary | flag | 0 | 4 | 0% | 50% |
| open_source | ×1.1 | 1 | 0 | 17% | 0% |
| salary_floor | cut | 0 | 3 | 0% | 38% |
| upfront_fee | cut | 0 | 1 | 0% | 12% |

Models only really differ on real, mostly non-English vacancies; the synthetic set is too small and too clean.

## Privacy

Nothing about a particular person is in the code: resumes, terms, vacancies, labels and the tracker are your files,
kept outside the repo. Data files are git-ignored, and `scripts/privacy_check.py` fails CI on real emails, phone
numbers and Telegram handles. `scripts/install-hooks.sh` adds the same check as a pre-commit hook, extended with your
own list of exact private strings kept outside the repo.

## Development

```bash
uv run pytest            # fast tests, hashing embedder
uv run pytest -m model   # one test with the real model
```
