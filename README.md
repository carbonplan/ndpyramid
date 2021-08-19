<img
  src='https://carbonplan-assets.s3.amazonaws.com/monogram/dark-small.png'
  height='48'
/>

# python-project-template
A carbonplan template for a developing a python project


![MIT License][]

[mit license]: https://badgen.net/badge/license/MIT/blue


This carbonplan repository contains a template for developing a python project. To start, click on the green [Use this template](https://github.com/carbonplan/python-project-template/generate) in the top right. This will allow you to create a new project using this base template.

## Modifications

### Updating project name

`scripts` and `tests` contain filler .py files. Update/remove these with your project name.
### Updating workflows/main.yaml

In the workflows/main.yaml file, the pytest and docker sections of the github actions configuration are currently commented out. If you wish to add them, uncomment them.

### Updating requirements.txt

requirements.txt is currently empty. You can populate it with: ```pip3 freeze > requirements.txt```

## license

All the code in this repository is [MIT](https://choosealicense.com/licenses/mit/) licensed, but we request that you please provide attribution if reusing any of our digital content (graphics, logo, articles, etc.).

## about us

CarbonPlan is a non-profit organization working on the science and data of carbon removal. We aim to improve the transparency and scientific integrity of carbon removal and climate solutions through open data and tools. Find out more at [carbonplan.org](https://carbonplan.org/) or get in touch by [opening an issue](https://github.com/carbonplan/research/issues/new) or [sending us an email](mailto:hello@carbonplan.org).
