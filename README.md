# JobSieve

**Rank job postings by whether you'd actually take them, not just by how much they look like your resume.**

Collect vacancies from public job APIs, rank them against your resumes, then apply your own terms — deal-breakers,
wishes, salary floor, red lines — as a second, explained stage. Learn from every fit / no-fit decision, see what
happens after you apply, and measure all of it on real decisions instead of eyeballing.

> **Status: done.** It runs my own job search. Use it as is, or fork it and bend it to yours; no further development
> is planned.

## Why

I built it for my own search. The first version was the obvious one: embed the resume, embed each vacancy, sort by
similarity on a 0–100 scale. It kept missing jobs I wanted and surfacing ones I would never take — roles that matched
my skills on paper and failed on pay, format, hours or plain honesty. Two changes fixed most of it: matching against
resume chunks instead of the whole resume, and a second stage that applies my own terms and says why.

The architecture, the calls and the mistakes are mine. AI did the typing.

```
## 4. Python Developer (client-facing) — CallStack
- score 30 · fit 60 · track A-backend · $2600 per month
- Why: Projects: Data pipeline (0.82); Summary (0.81)
- Adjusted: language ×0.5
- Similar labeled: fit 0.94 Python Backend Developer; fit 0.92 Telegram Bot Developer; no_fit 0.92 Call Center Agent; ...
```

The vacancy matches the resume (fit 60), but it needs fluent German, which this candidate does not speak. The score
halves and the job drops below the ones they would actually take. Every number on the card says where it came from.

## What it is not

- **Not an auto-apply bot.** It ranks and explains; you decide and send.
- **Not a scraper.** Vacancies come from the boards' own public APIs or pages you open yourself. No logins.
- **Not a cloud service.** The model runs locally on CPU with no API key; resumes, terms and decisions stay in files
  on your machine.
- **Not right on day one.** Stage 2 is only as good as your terms, and the numbers only as good as your labels — see
  [On a real search](#on-a-real-search).

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

   **One scale for all languages, or one per language.** The model places whole languages at different heights:
   against the same resumes, Russian vacancies sat a full scale-width above English ones. Part of that is the model,
   part can be you — a candidate who can't work in a language really does fit vacancies in it less often. So the
   default is one scale, and `vm calibrate --per-language` gives each language with 30+ vacancies a scale of its own
   (the language is guessed without extra libraries: by script, and by frequent words for Latin-script languages).
   Once you have some fit/no_fit labels (`vm label`), measure before you switch: run `vm eval` after each
   `vm calibrate` and keep the one with the higher AUC. On a real base the shared scale won — AUC 0.59 against 0.34 on a run of 148 vacancies — because the roles in one
   language there were mostly ones the candidate could not take. Per-language scales suit a base where fits are spread evenly across
   languages; they give each language's median vacancy 50 points, as if every language were equally relevant. The
   choice is kept in the base, through a new profile or model, until a `vm calibrate` without the flag; a language
   under 30 vacancies uses the shared scale, and its fits move once when it reaches 30.

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
one with none. A cut sends a vacancy below every other one whatever its fit. Boosts can lift a score above 100.

### Feedback from your decisions

Labeled vacancies vote: a new vacancy close to three rejected scams moves down, one close to jobs you took moves up —
by up to 15 points of fit. The vote switches on once there are at least 100 labels, 20 of them rejections; until
then it would mostly echo a handful of decisions.

## Your terms: wishes, conditions, red lines

Similarity answers "does this look like my experience?". It does not answer "would I take it?". A job that needs
daily calls in a language you do not speak, pays half your floor, or is a scam can look like a perfect match on paper
and land right next to the jobs you want. Examples of terms people set: *no client calls in French — I read it
fine, can't hold a conversation* (a condition); *no unpaid test tasks longer than a day* (a red line).

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
  on_call: ['on-?call', '24/7 support']
conditions:
  language: ['fluent german', 'german \(?c1', 'native german']
red_lines:
  upfront_fee: 'registration fee'      # one pattern may be a plain string
salary:   # $1500 a month and up no effect, $1350 ×0.9, $1200 ×0.65, $1100 ×0.5, below $1100 cut
  currency: USD
  period: month
  curve: {1500: 1.0, 1350: 0.9, 1200: 0.65, 1100: 0.5}
  rates: {USD: 1.0, EUR: 1.08, GBP: 1.27}
```

The whole file: [`examples/rules.example.yaml`](examples/rules.example.yaml). `levels:` changes the tier defaults;
any rule can set its own, e.g. `office: {patterns: ['office'], factor: 0.7}`.

- **Patterns** are case-insensitive regular expressions. Write them in single quotes: inside YAML double quotes `\b`
  turns into a backspace. Mistakes — an unknown section such as `wishs:`, a broken regex, a YAML error — stop the
  command with one line that names them, before the model loads.
- **The salary curve** is written in your own currency and period: `currency: EUR`, `period: year`,
  `curve: {60000: 1.0, 48000: 0.5}`. `rates` say how many USD one unit of a currency is worth (`EUR: 1.08`,
  `KZT: 0.002`) — list every currency vacancies may be paid in, USD is always 1; a rate written the other way round
  (`KZT: 450`) is refused. Cards show the salary as the vacancy writes it; the `Adjusted:` line shows the figure
  the curve read, in USD a month.

- **Why a condition halves instead of cutting:** a regex misreads ("German is a plus"), so a serious miss costs half
  and a near-perfect vacancy still surfaces with the reason next to it. Cuts are for what is never worth reading.
- **Salary:** a range counts at its middle (1000–1400 → 1200). Part-time is read from the title, the work format
  and the salary line — a body says "this is not part-time" as often as "part-time". Its pay is scaled to full time
  when the hours are stated anywhere (20 h/week at $900 ≈ $1660 full time); without hours it is flagged
  (`part_time_pay`) and not judged. Hourly
  pay counts 160 hours a month. A figure with no period that would be over $15,000 a month is read as a yearly salary:
  US and EU boards rarely write "per year". Pay per project is flagged, not judged against a monthly floor. An
  unknown salary, or a currency missing from `rates:`, is flagged (`no_salary`), never penalised: ask.
- **Tuning:** the factors are starting guesses. `vm eval` reports how often each rule fired on vacancies you took and
  on ones you rejected; a penalty that hits as many fits as no-fits is too strict.

## Quick start

Needs [uv](https://docs.astral.sh/uv/); it fetches Python 3.13 itself. About 2 GB of disk: the environment with
PyTorch is ~0.8 GB, the default model 1.1 GB downloads on first use (15+ minutes on a slow line, with no progress
bar) and is read from the cache after that. No API key: Hugging Face prints a warning about `HF_TOKEN`, it is not
needed. For a lighter start add `--model intfloat/multilingual-e5-small` (~0.5 GB, a little less accurate) to every
command — `--model` is a global option, and one base should stay on one model.
On Windows the commands are the same in PowerShell; set variables with `$env:VM_DB = "my.db"` instead of `export`.
The package installs as `vacancy-match` and the command is `vm`.

```bash
git clone https://github.com/Augustforge/Forge_JobSieve && cd Forge_JobSieve
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
   (`resume-backend-EN.md`, `resume-backend-ES.md`, `resume-data-DE.md`), all in one folder. `<LANG>` is a two-letter
   language code; the same direction in two languages is one track, so don't end a track name with a language code
   (`resume-support-IT.md` reads as Italian; AI, ML, BI, HR stay part of the track name). `index-profile` prints the tracks it found — check them. Each
   `## Section` becomes a chunk, and inside a section a line that is bold as a whole (`**Order bot**`) starts a
   project chunk. Above the first `##` only `###` lines (your title) are kept, as a chunk named `header` — name and
   contacts are skipped. A file without `##` is indexed as one chunk, with a
   note. `index-profile` replaces the whole profile, so
   index the folder, not one file.
2. **Your terms (optional).** Copy `examples/rules.example.yaml` to `private/my-rules.yaml` and write your own salary
   curve, conditions, wishes
   and red lines. Skip it and the ranking is resume fit alone.
3. **Vacancies.** `vm fetch` collects them from the boards' own public APIs — no scraping, no login — into JSONL,
   one run per file (see the table below). Any other source works too: JSONL with `source`, `url`, `title` and
   ideally `company`, `text`, `salary_text`, `published_at`, `work_format` (full-time, part-time, project) and
   `location`. Every run has an id with a date (`2026-10-14`, `links-2026-10-14`) — runs are replayed in date order.
   `vm fetch` stamps today's date (or `--run`); for your own JSONL pass `vm add file.jsonl --run 2026-10-14`.
4. **Run it:**

```bash
mkdir private                                      # git-ignored: resumes, rules, base and outputs stay yours
export VM_DB=private/my.db VM_RULES=private/my-rules.yaml   # VM_RULES only if you wrote rules; or --db / --rules
uv run vm index-profile private/resumes
uv run vm fetch himalayas:python remoteok:python greenhouse:gitlab --where 'python|backend' --out private/run.jsonl
uv run vm add private/run.jsonl                    # prints the run id fetch stamped: today's date
uv run vm calibrate                                # once you have 50+ vacancies (add --per-language if you chose it)
# uv run vm calibrate --per-language               # optional, compare with vm eval first: How the score works, 4
uv run vm rank --run 2026-10-14 --out private/top.md   # the run id from the line above; top 50, --top N for more
uv run vm label <url> fit --track backend          # every decision, fit and no_fit
uv run vm label <url> no_fit --reasons onsite,cold-calls   # reasons are your own words, one word each
uv run vm eval                                     # after a few runs with real rejections
```

### Where `vm fetch` collects from

| source | what | argument |
|---|---|---|
| `greenhouse:<board>` | one company's open jobs on Greenhouse | the name in `job-boards.greenhouse.io/<board>` |
| `lever:<company>` | one company's open jobs on Lever | the name in `jobs.lever.co/<company>` |
| `ashby:<org>` | one company's open jobs on Ashby | the name in `jobs.ashbyhq.com/<org>` |
| `remotive:<search>` | remote jobs, Remotive | a search, e.g. `python` — the free API returns the same small sample whatever the search |
| `remoteok[:<tag>]` | remote jobs, Remote OK | optional tag, e.g. `python` |
| `himalayas:<search>` | remote jobs, Himalayas | a search |
| `arbeitnow` | jobs in Germany and remote, Arbeitnow | — |
| `hh:<search>` | HeadHunter (hh.ru, hh.kz, …) | needs `HH_TOKEN`: since 2026 the hh API answers only registered apps ([dev.hh.ru](https://dev.hh.ru)) |
| `page:<url>` | one vacancy page you found yourself | any page with schema.org `JobPosting` markup — hh vacancy pages have it, no token needed |

`--where REGEX` keeps jobs whose title, location or work format match, ignoring case. Filter by title
(`--where 'python|backend'`): aggregators write the location as "Worldwide" or a region, not "remote", and their
searches are loose — Remote OK's `python` tag returns store managers too. A source that fails
(a typo in a board name, no network) is reported and the others are still collected. Every record keeps the job's
own URL and its source: the aggregators ask you to link back and credit them if you republish their jobs, and
Remotive asks for no more than a few requests a day.

## After you apply: the funnel

Ranking decides what to send; the funnel shows what happened to it. Keep a CSV tracker
([`examples/tracker.example.csv`](examples/tracker.example.csv)):

| column | meaning |
|---|---|
| `company` | required |
| `sent` | `2026-10-05` or `05.10.2026`, required unless the row is a draft |
| `status` | `draft` (not confirmed sent) · `sent` · `replied` · `interview` · `offer` · `rejected` · `closed` (you stopped); Russian words such as `отправлено`, `отказ` work too |
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
- **Track accuracy**, the worst misses both ways (with their score, 0–100 as in `vm rank`), and the per-rule report.

With a couple of dozen labels the interval is wide (0.5–0.95 is common): two settings whose intervals overlap are
not yet different. Comparing one scale with per-language scales takes labels in each language you rank.

It counts your own decisions by default: an agent's labels would only measure the system agreeing with itself
(`--labels all` includes them).

### On a real search

My own base, October 2026: 402 vacancies with text from 5 fetch runs (12–16 Oct) plus earlier applications.
322 labels, but only 14 are my own hand labels (11 fit, 3 no-fit). The other 308 were seeded from my application
tracker, with "I sent it" counted as fit.

| measure | stage 1 | stage 1+2 |
|---|---|---|
| AUC on my 14 hand labels [95% CI] | 0.70 [0.42–0.94] | 0.79 [0.48–1.00] |
| share of fits in the top 10, all runs | 0.72 | 0.82 |
| AUC across runs, all labels | 0.43 | 0.48 |

- **Inside a run the order works.** On seeded labels within single runs, AUC is 0.83 (20 labeled vacancies) and 0.69
  (45 labeled out of 148).
- **Across runs it is a coin flip.** The likely cause is the seed: "sent" is not "best match", since I also applied
  to so-so fits, and different sources write very different texts.
- **14 hand labels prove nothing.** The interval runs from a coin flip to perfect. The number starts to mean
  something at 50+ hand labels with 15+ rejections.
- **The rule report earned its place.** Stale postings (older than 14 days) hit 26% of no-fits and 4% of fits. A ×1.1
  crypto boost I had set lifted more vacancies I rejected (56%) than ones I took (36%): the tool caught my own bad rule.

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
kept outside the repo or in `private/`, which git ignores. Data files are git-ignored too, and `scripts/privacy_check.py` fails CI on real emails, phone
numbers and Telegram handles. `scripts/install-hooks.sh` adds the same check as a pre-commit hook, extended with your
own list of exact private strings kept outside the repo.

## Development

```bash
uv run pytest            # fast tests, hashing embedder
uv run pytest -m model   # one test with the real model
```

---

Part of **[Augustus Forge](https://x.com/augustus_forge)**: autonomous systems, built in public. Failures included.
Wins earned.
