"""The desktop window of 11footy, for anyone who downloads the repository: Vesta's forecasts on
a dataset, computed on their own PC.

`python -m desk` (or `11footy`, or a double-click on a launcher: `11footy.bat`,
`11footy.command`, `11footy.sh`) opens one window. It opens a dataset file, ours (refreshed
after each round) or anyone's in the Vesta data format (`desk.opening`), runs Vesta's command
on it (`desk.run`), and shows what Vesta wrote in the dataset's own forecast log on four
screens (`desk.reading`): the round of a competition, the whole forecast of one match, the
season tables and the coverage. No source is fetched, nothing reaches the network, and no
probability is computed in the window itself. Python, Tk, the Sun Valley theme and the
numerical libraries Vesta needs are its whole install. Percentages only.
"""
