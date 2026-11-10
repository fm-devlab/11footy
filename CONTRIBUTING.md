# Contributing to 11footy

Thank you for helping. Issues are welcome, and so are small pull requests.

## Where to write

- **A bug** in the window or the code: open an issue with the "Bug report" form.
- **A dataset file the window cannot read**: the "Dataset problem" form. Wrong football
  facts (a score, a scorer, a date) belong in the 11footy-dataset repository.
- **A question or an idea**: Discussions.
- **A security problem**: never in public; see `SECURITY.md`.

## Pull requests

- One topic per pull request, kept small.
- The tests pass on your PC: `uv sync --extra dev`, then `uv run python -m pytest tests -q`.
- Sign off every commit (`git commit -s`). This certifies the Developer Certificate of Origin
  (https://developercertificate.org): you have the right to submit the work.

## The licence of your contribution

11footy is source-available under the PolyForm Noncommercial License 1.0.0 (`LICENSE`). It is
free for non-commercial use. It is not an OSI-approved licence.

By opening a pull request you agree that:

1. your contribution is licensed under the same PolyForm Noncommercial License 1.0.0; and
2. 11footy may also license your contribution under other terms, including commercial licences
   of the code.

If you do not agree, open an issue that describes the change instead of a pull request.

## Ground rules

- Be kind and stay on topic.
- 11footy publishes probabilities for fans and analysts. Gambling content and links to
  gambling services are removed.

For licence questions, or anything that does not belong in a public issue: `11footy.game@gmail.com`.
