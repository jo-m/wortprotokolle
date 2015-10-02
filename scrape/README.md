# Scrape Wortprotokolle
Skript, um die [Wortprotokolle](http://www.parlament.ch/d/dokumentation/Seiten/default.aspx) des amtlichen Bulletins zu scrapen. Dient dazu, Text-Modelle zu trainieren. Es wird ein Node-paket benötigt:

    npm install html-to-text

Benutzung: Sessions-Nummer herausfinden im Inhaltsverzeichnis (z.B. 4920). Danach:

    ./scrape.sh 4920
