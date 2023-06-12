## Plot conditional means

Follow `exec.sh`

### Compute conditional means

```
usage: compute_conditionalMean.py [-h] -f  [-vert] [-avg] [-fl FIELD_LIST [FIELD_LIST ...]]

Case folder

options:
  -h, --help            show this help message and exit
  -f , --caseFolder     caseFolder to analyze
  -vert , --verticalDirection 
                        Index of vertical direction
  -avg , --windowAve    Window Average
  -fl FIELD_LIST [FIELD_LIST ...], --field_list FIELD_LIST [FIELD_LIST ...]
                        List of fields to plot
```

### Plot conditional means

```
usage: plot_conditionalMean.py [-h] [-cf  [...]] [-fl  [...]] [-ff] [-n  [...]]

Plot conditional means

options:
  -h, --help            show this help message and exit
  -cf  [ ...], --caseFolders  [ ...]
                        caseFolder to analyze
  -fl  [ ...], --field_list  [ ...]
                        fields to analyze
  -ff , --figureFolder 
                        figureFolder
  -n  [ ...], --names  [ ...]
                        names of cases
```
