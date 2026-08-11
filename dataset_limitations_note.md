# Dataset limitations — for the Methodology / Limitations section

Use as-is, or adapt the wording to match the rest of the chapter.

---

A limitation of the spatial variables in this study concerns how each
listing's borough was assigned. Listings were collected via 33 separate
borough-level Airbnb searches, and the borough recorded against each
listing (`search_borough`) reflects which search returned it rather than
a verified geographic location. Because Airbnb's search returns all
listings within a radius of the queried area, and that radius crosses
administrative boundaries, a listing can be returned by more than one
borough's search — including, near the edge of Greater London, searches
whose radius extends into neighbouring counties. `search_borough` itself
cannot be used to detect this, since it only ever holds the 33 queried
borough strings; instead, each listing's title (which Airbnb renders as
"<type> in <place>", e.g. "Room in Croydon") was parsed for its trailing
place name, and rows naming a place outside Greater London (Surrey,
Essex, Hertfordshire, Berkshire, Buckinghamshire, Kent, Thurrock, and
specific Home Counties towns such as Windsor, Esher, and Chigwell) were
dropped prior to modelling. This removed 205 of 2,725 rows (7.5%),
concentrated in the boroughs that border those counties — chiefly
Hillingdon, Havering, Sutton, Croydon, Hounslow, and Harrow. Additionally,
where the same listing was returned by more than one borough's search, it
is retained under only one borough in the final dataset (the first
encountered, in alphabetical order of search), which means the recorded
distribution of listings across boroughs — and by extension the
Inner/Outer London classification derived from it — is subject to a
degree of systematic, non-random measurement error rather than pure
sampling noise. This is most likely to understate density in boroughs
bordering many others, such as Newham, Tower Hamlets, and the City of
London. A more precise approach would assign each listing's borough from
its own geographic coordinates via point-in-polygon matching against an
Office for National Statistics boundary file; this was judged out of
scope given the project timeline and is noted here as a direction for
future refinement of the dataset.

A related limitation concerns sampling depth. To avoid duplicate records,
data collection for a given (borough, price band) search stopped as soon
as a subsequent page returned no listings not already seen. In practice,
most searches stopped after the first page: 2,397 of the 2,520 listings
in the cleaned dataset were collected from page 1 of their respective
search. Since Airbnb orders search results by its own relevance ranking
rather than randomly, the dataset should be understood as a sample of
highly-ranked listings per search rather than a random sample of London's
full Airbnb supply. This does not affect the internal validity of the
collected records, but it means the sample should not be described as
representative of the London Airbnb market as a whole, and this
limitation is acknowledged accordingly.
