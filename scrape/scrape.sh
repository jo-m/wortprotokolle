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

get_text() {
  session=$1
  echo > $session.txt
  find "www.parlament.ch/ab/data/d/n/$session/" -name '*.htm' | sort | while read f; do
    node_modules/html-to-text/bin/cli.js < $f | \
      sed 's/\[.*\]//' | \
      egrep -v '^ Home' >> $session.txt
  done
}

scrape $1
get_text $1

# URLs
# http://www.parlament.ch/ab/toc/d/n/4920/d_n_4920.htm
# http://www.parlament.ch/ab/toc/d/n/4920/474367/d_n_4920_474367.htm
# http://www.parlament.ch/ab/frameset/d/n/4920/474367/d_n_4920_474367_474368.htm
# http://www.parlament.ch/ab/frameset/d/n/4920/474367/d_n_4920_474367_474368.htm
# http://www.parlament.ch/ab/data/d/n/4920/474367/d_n_4920_474367_474368.htm <- Daten hier
