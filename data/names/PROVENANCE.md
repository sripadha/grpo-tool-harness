# Where the name lists come from

`first_names.txt` and `last_names.txt` hold common first names and common surnames,
meant to be combined into made-up person names. Both lists come from files published
by the US Census Bureau. This page says which files, when they were downloaded, how the
names were chosen, and what the Census Bureau says about the files and about reuse.

The same facts are in `provenance.json`, which also lists the SHA-256 of every saved page.

All times are US Pacific Standard Time (UTC-08:00), from the clock of the machine that
made the requests. Requests were made with curl 7.81.0 sending the user agent `curl/8`:
no cookies, no login, no personal data. Raw downloads are kept in `.cache/names_raw/`.
`.gitignore` lists `.cache/`, so they are not part of the repository.

## The three raw files

### 1. Female first names

| Fact | Value |
|---|---|
| Source | US Census Bureau: Frequently Occurring Names from the 1990 Census, file dist.female.first |
| Downloaded from | https://www2.census.gov/topics/genealogy/1990surnames/dist.female.first |
| HTTP status | 200 |
| Downloaded at | 2026-11-17T00:00:51-08:00 (2026-11-17T08:00:51Z) |
| Kept as | `.cache/names_raw/census_1990_dist.female.first` (149625 bytes, 4275 lines) |
| SHA-256 | `bd2f310fc4e5d5e5ea122c9d4342c9821145823118eb20db1647f305ec77b358` |

### 2. Male first names

| Fact | Value |
|---|---|
| Source | US Census Bureau: Frequently Occurring Names from the 1990 Census, file dist.male.first |
| Downloaded from | https://www2.census.gov/topics/genealogy/1990surnames/dist.male.first |
| HTTP status | 200 |
| Downloaded at | 2026-11-17T00:00:51-08:00 (2026-11-17T08:00:51Z) |
| Kept as | `.cache/names_raw/census_1990_dist.male.first` (42665 bytes, 1219 lines) |
| SHA-256 | `0a5078ef6effe3b483d15b0f7f95047662126c9bfb624ecd5e5b978fc0f2470b` |

Both first-name files are plain text. Each line holds a name in capitals, its frequency
in percent, the cumulative frequency in percent, and the rank. The lines are in rank
order. For both files the server sent `Last-Modified: Thu, 31 Jul 2014 14:15:52 GMT`.

### 3. Last names

| Fact | Value |
|---|---|
| Source | US Census Bureau: Frequently Occurring Surnames from the 2010 Census (File B: Surnames Occurring 100 or more times) |
| Downloaded from | https://www2.census.gov/topics/genealogy/2010surnames/names.zip |
| HTTP status | 200 |
| Downloaded at | 2026-11-16T23:39:56-08:00 (2026-11-17T07:39:56Z) |
| Kept as | `.cache/names_raw/census_2010_surnames_names.zip` (12874389 bytes) |
| SHA-256 | `117c41cb4668727b7627b2845b6df3f83eb2a22a1813f42c0ff4bdcab86de135` |
| Files inside the archive | `Names_2010Census.csv`, `Names_2010Census.xlsx` |
| File used | `Names_2010Census.csv` (9445924 bytes) |
| SHA-256 of the file used | `6b4933178060b4e5ac8dd9a611ec15472d35eab98fe9027474bc5ccc5b7a8708` |

The file used has one header line
(`name,rank,count,prop100k,cum_prop100k,pctwhite,pctblack,pctapi,pctaian,pct2prace,pcthispanic`),
then 162,253 surnames in rank order, then a last row `ALL OTHER NAMES`.
The server sent `Last-Modified: Thu, 15 Dec 2016 15:21:36 GMT`.

## The two lists

| File | Lines | SHA-256 | Made from |
|---|---|---|---|
| `first_names.txt` | 200 | `be99b057c35573b35dd4c5f352846490c22c7417abdf06fe447d1793bd05bd9c` | raw files 1 and 2 |
| `last_names.txt` | 200 | `785f1a001034646c828027893ac29575224eb588d4734403224dc421e8f4c362` | raw file 3, leaving out the 200 first names |

`first_names.txt` begins: Mary, James, Patricia, John, Linda, Robert, Barbara, Michael,
Elizabeth, William, Jennifer, David.

`last_names.txt` begins: Smith, Johnson, Williams, Brown, Jones, Garcia, Miller, Davis,
Rodriguez, Martinez, Hernandez, Lopez.

## What the first names represent

The 1990 files tabulate the names of people recorded in the 1990 census. They are not a
list of babies born in a given year. So the first names are those of the generations
living in 1990, which today means older generations: the lists begin Mary, Patricia,
Linda and James, John, Robert.

## How the names were chosen

### First names

1. Walk dist.female.first and dist.male.first in rank order. This is the order of each
   file; the script stops if a rank ever goes down.
2. Skip a name with fewer than 3 letters.
3. Skip a name with a character outside A-Z.
4. Skip a name that is taken under the other sex. A name that is in both files is
   considered only under the sex where its frequency in percent is higher (if the two
   are equal: female).
5. Take the first 100 female names and the first 100 male names that pass.
6. Write each name with a capital first letter and the rest in lower case
   (MARY becomes Mary).
7. Write `first_names.txt` interleaved: female 1, male 1, female 2, male 2, and so on
   (200 lines).

Example for rule 4: TERRY is in both files, with 0.311 percent in the male file
and 0.080 percent in the female file, so TERRY can only be chosen as a male name.

### Last names

1. Walk `Names_2010Census.csv` in rank order. This is the order of the file; the script
   stops if a rank ever goes down.
2. Skip the row `ALL OTHER NAMES`.
3. Skip every name that starts with MC. Plain capitalisation would spoil it:
   MCDONALD would become Mcdonald.
4. Skip a name with a character outside A-Z.
5. Skip a name that is equal, ignoring upper and lower case, to any of the 200 chosen
   first names.
6. Write each name with a capital first letter and the rest in lower case
   (SMITH becomes Smith).
7. Take the first 200 names that pass and keep them in rank order.

### What was skipped

First names:

| Reason | Names skipped |
|---|---|
| fewer than 3 letters | 0 |
| character outside A-Z | 0 |
| in both files and belongs to the other sex | 0 |

Nothing was skipped. The 100 names taken from each file are its ranks 1 to 100.
331 names are in both files, but none of them is within the first 100 of both, and each
one within the first 100 of a file has its higher frequency there.

Last names:

| Reason | Names skipped | Which (rank in the 2010 file) |
|---|---|---|
| the row ALL OTHER NAMES | 0 | the walk ended before that row |
| starts with MC | 1 | MCDONALD (143) |
| character outside A-Z | 0 | |
| same as a chosen first name | 10 | THOMAS (16), MARTIN (20), SCOTT (36), KELLY (74), HOWARD (75), JAMES (85), RUSSELL (104), HENRY (149), ROSE (177), RYAN (193) |

11 surnames were skipped inside the walk, so the 200th surname taken is Cunningham at
rank 211 (200 + 11).

Checked on the 2010 file: Ranks 1 to 211: every rank occurs once and the ranks rise in
file order; no name starts with MAC; MCDONALD is the only name starting with MC.

### Checks

| Check | Result |
|---|---|
| Lines in `first_names.txt` | 200 |
| Lines in `last_names.txt` | 200 |
| A name twice inside `first_names.txt`, ignoring case | no |
| A name twice inside `last_names.txt`, ignoring case | no |
| A name in both files, ignoring case | none |
| Two runs give byte-identical files | yes |
| Script uses only the Python standard library | yes |
| Script reads only the three raw files | yes |

The script was run three times on 2026-11-17; the two lists were byte-identical each time.
The script imports csv, decimal, hashlib, io, json, pathlib, string, sys and zipfile.
In a watched run it opened three files for reading, the three raw files in
`.cache/names_raw/`, and two files for writing, the two lists in `data/names/`.

## What the Census Bureau says about the 1990 files

https://www.census.gov/topics/population/genealogy/data/1990_census.html
("Frequently Occurring Names from the 1990 Census", "Page Last Revised - November 13, 2026",
fetched 2026-11-17T00:01:08-08:00, HTTP 200):

> NOTE: No specific individual information is given.

> The Census Bureau receives numerous requests to supply information on name frequency. In an effort to comply with those requests, the Census Bureau has embarked on a names list project involving a tabulation of names from the 1990 Census. These files contain only the frequency of a given name, no specific individual information.

https://www.census.gov/topics/population/genealogy/data/1990_census/1990_census_namefiles.html
("Frequently Occurring Names from Census 1990 Names Files", "Page Last Revised - November 13, 2026",
fetched 2026-11-17T00:01:24-08:00, HTTP 200). This page links dist.all.last,
dist.female.first and dist.male.first, at the addresses used above:

> NOTE: No specific individual information is given.

> Each of the three files, (dist.all.last), (dist. male.first), and (dist female.first) contain four items of data. The four items are:

> A "Name" Frequency in percent Cumulative Frequency in percent Rank

> In our search area sample, MOORE ranks 9th in terms of frequency. 5.312 percent of the sample population is covered by MOORE and the 8 names occurring more frequently than MOORE. The surname, MOORE, is possessed by 0.312 percent of our population sample.

On the page the third quote is four list items, one per line.

https://www2.census.gov/topics/genealogy/1990surnames/nam_meth.txt (the document both
pages link as "Name List: Documentation and Methodology"; its heading is
"DOCUMENTATION AND METHODOLOGY FOR FREQUENTLY OCCURRING NAMES IN THE U.S.--1990";
fetched 2026-11-17T00:01:24-08:00, HTTP 200):

> The PES blocks plus the surrounding ring--"the Search Area"--contained 7.2 million census records replete with name. It is this Search Area data set that provides the impetus for the three name files at this internet site.

> For purposes of both confidentiality and elimination of data noise we restricted the number of unique names available at this internet site to the minimum number of entries that contain 90 percent of the population in that data file.

> The PES sample intentionally over sampled both Blacks and Hispanics, and it is likely that the Search Area also contains an excess of these two groups. Thus the frequently occurring surnames: GARCIA, RODRIGUEZ, GOMEZ and WASHINGTON as well as first names: JUAN, JOSE, GUADALUPE and WILLIE might attain higher rankings than their actual population numbers within the United States would warrant.

> Selected data from these files have appeared in the print media with the citation "source--Census Bureau".

In the document the quoted sentences run over several lines; line breaks are written
here as single spaces. PES stands for the Post-Enumeration Survey of 1990. Juan, Jose
and Willie are among the 200 chosen first names; Guadalupe is not.

Numbers the document gives. Valid records and unique names: dist.female.first 3,184,399
and 4,275; dist.male.first 3,003,954 and 1,219 (printed there as 1.219). The downloaded
files have 4,275 and 1,219 lines.

## Reuse statements, quoted word for word

**The pages of the 1990 files: no explicit statement found on the page**
(https://www.census.gov/topics/population/genealogy/data/1990_census.html and
https://www.census.gov/topics/population/genealogy/data/1990_census/1990_census_namefiles.html),
and none in the linked documentation
(https://www2.census.gov/topics/genealogy/1990surnames/nam_meth.txt). Their text was
searched for: public domain, copyright, reuse, re-use, permission, license, licence,
free of charge, restriction, freely, cite, citation, terms of. The only matches are
'in terms of frequency' and the sentence about the citation 'source--Census Bureau'
quoted above. That sentence reports how others credited the data; it is not a permission.

**The page of the 2010 file: no explicit statement found on the page**
(https://www.census.gov/topics/population/genealogy/data/2010_surnames.html,
"Page Last Revised - October 8, 2021", fetched 2026-11-16T23:44:00-08:00, HTTP 200).
The text of the page was searched for the same words. None occurs. The page does say:

> NOTE: This presentation of data focuses on summarized aggregates of counts and characteristics associated with surnames, and the data do not in any way identify any specific individuals.

The general rule on the Census Bureau's own site:

- https://www.census.gov/about/policies/open-gov.html ("Open Government at the Census
  Bureau", "Page Last Revised - July 8, 2026", fetched 2026-11-16T23:45:33-08:00, HTTP 200):

  > The U.S. Census Bureau publishes its data as open data, meaning it is freely available for use and re-use by the public.

- https://www.census.gov/topics/research/research-transparency-public-access/open-data.html
  ("Open Data", "Page Last Revised - July 8, 2026", fetched 2026-11-16T23:44:32-08:00,
  HTTP 200; reached from https://www.census.gov/about/policies/open-gov/open-data.html,
  which redirects):

  > The U.S. Census Bureau is committed to open government by sharing its public data as open data.

- https://www.census.gov/about/policies/citation.html ("Citing our Data, Tools, Technical
  Documents and Research", "Page Last Revised - February 24, 2026", fetched
  2026-11-16T23:44:31-08:00, HTTP 200). This asks for the Census Bureau to be named as
  the source. It is not a licence:

  > Data users who create their own estimates using data from disseminated tables and other data should cite the Census Bureau as the source of the original data only.

The general rule in US law, from the US Copyright Office's text of the copyright law,
section 105(a) (https://www.copyright.gov/title17/92chap1.html, fetched
2026-11-16T23:46:23-08:00, HTTP 200):

> Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise.

## What happened with the Social Security Administration file

The first choice for first names was the national baby-names file of the US Social
Security Administration (https://www.ssa.gov/oact/babynames/names.zip). The site refused
automated download: on 2026-11-16 it answered HTTP 403 'Access Denied' to five requests
made with curl between 23:36:41 and 23:56:35, and also to requests for its describing
page, its home page and its robots.txt. The data.gov catalog lists the same address and
holds no copy. No copy from another site was used. The Census Bureau's 1990 first-name
files were used in its place.

## Suggested credit line

> Name lists derived from U.S. Census Bureau data: Frequently Occurring Names from the 1990 Census (first names) and Frequently Occurring Surnames from the 2010 Census (last names).

Fuller citations, following the pattern on the Census Bureau's citation page
(https://www.census.gov/about/policies/citation.html), which reads
"Author/s, “Table or File Name/Number,” Full Product Name, Vintage, <URL>, accessed on date.":

- U.S. Census Bureau, "dist.female.first" and "dist.male.first," Frequently Occurring Names from the 1990 Census, <https://www.census.gov/topics/population/genealogy/data/1990_census.html>, accessed on November 17, 2026.
- U.S. Census Bureau, "Names_2010Census.csv," Frequently Occurring Surnames from the 2010 Census, <https://www.census.gov/topics/population/genealogy/data/2010_surnames.html>, accessed on November 16, 2026.

The credit line and the filled-in citations are suggestions written for this project;
only the pattern is the Census Bureau's wording.

## Rebuild

`make_name_lists.py` uses the Python standard library only.

    python3 -I data/names/make_name_lists.py

It reads the three raw files in `.cache/names_raw/`, writes `data/names/first_names.txt`
and `data/names/last_names.txt`, and prints the SHA-256 of the three raw files, of the
file used inside the archive and of the two lists, the names skipped under each rule,
and the checks.

To repeat from nothing: download the three raw files from the addresses above into
`.cache/names_raw/` under the names given above, compare their SHA-256 with this page,
then run the command.
