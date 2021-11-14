#!/usr/bin/env bash

tpu_name=$1

ctpu up --name=${tpu_name} --project=neulab --tf-version=2.3 --tpu-size=v3-8 --tpu-only --noconf
