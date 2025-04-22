# Motivation

## Description

Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience and huge economic losses. As a result, there is growing interest in predicting flight delays to optimize operations and improve customer satisfaction. In this project, we will predict flight delays using the DOT flights and NOAA weather datasets from 2015 to 2021.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://mids-w261/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://mids-w261/data/` to `data/`.


