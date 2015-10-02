#!/bin/bash

scrape() {
  session=$1
  root='http://www.parlament.ch/ab/static/ab/html/d/default.htm'
  wget --continue \
       --recursive \
       --level=8 \
       --convert-links \
       --accept-regex="^http://www\.parlament\.ch/ab/(toc|frameset|data)/d/n/$session.*\$" \
       $root 2>&1
}

scrape 4920

# URLs
# http://www.parlament.ch/ab/toc/d/n/4920/d_n_4920.htm
# http://www.parlament.ch/ab/toc/d/n/4920/474367/d_n_4920_474367.htm
# http://www.parlament.ch/ab/frameset/d/n/4920/474367/d_n_4920_474367_474368.htm
# http://www.parlament.ch/ab/frameset/d/n/4920/474367/d_n_4920_474367_474368.htm
# http://www.parlament.ch/ab/data/d/n/4920/474367/d_n_4920_474367_474368.htm <- Daten hier
