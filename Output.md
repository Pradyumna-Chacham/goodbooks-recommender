#SAMPLE OUTPUT
```

⏳ Loading inference artifacts...
⏳ Building TF-IDF matrix for Content-Based Filtering...
⏳ Precomputing CF vectors (top-50 item neighbors)...
⏳ Loading RL Q-network weights...
✅ All artifacts loaded successfully!


==============================
📚 BOOK RECOMMENDER SYSTEM
==============================
1. Random user → compare all models
2. Enter a book title → compare Item-CF / SVD / RL / Hybrid / CBF
3. Enter a user ID → compare all models
4. Exit

Choose an option: 1

🎯 Random user selected: 42728

User's Top Rated Books:
-----------------------
  [1892] Case Histories (Jackson Brodie #1) — Kate Atkinson
  [1712] Still Life (Chief Inspector Armand Gamache, #1) — Louise Penny
  [6627] Pardonable Lies (Maisie Dobbs, #3) — Jacqueline Winspear
  [7978] The Mapping of Love and Death (Maisie Dobbs, #7) — Jacqueline Winspear
  [4401] A Rule Against Murder (Chief Inspector Armand Gamache, #4) — Louise Penny


🔎 Recommendations for user_id = 42728

User-CF Recommendations:
------------------------
  [840] Shōgun (Asian Saga, #1) — James Clavell
  [639] Heidi — Johanna Spyri, Angelo  Rinaldi, Beverly Cleary
  [1493] Old Yeller (Old Yeller, #1) — Fred Gipson
  [237] Carrie — Stephen King
  [364] How the Grinch Stole Christmas! — Dr. Seuss
  [799] Watchers — Dean Koontz
  [976] Dr. Seuss's Green Eggs and Ham: For Soprano, Boy Soprano, and Orchestra — Robert Kapilow, Dr. Seuss
  [337] The Ultimate Hitchhiker's Guide to the Galaxy — Douglas Adams
  [740] The Little House Collection (Little House, #1-9) — Laura Ingalls Wilder, Garth Williams
  [358] Oh, The Places You'll Go! — Dr. Seuss


Item-CF Recommendations:
------------------------
  [4] To Kill a Mockingbird — Harper Lee
  [33] Memoirs of a Geisha — Arthur Golden
  [11] The Kite Runner — Khaled Hosseini
  [31] The Help — Kathryn Stockett
  [5] The Great Gatsby — F. Scott Fitzgerald
  [22] The Lovely Bones — Alice Sebold
  [46] Water for Elephants — Sara Gruen
  [9] Angels & Demons  (Robert Langdon, #1) — Dan Brown
  [8] The Catcher in the Rye — J.D. Salinger
  [7] The Hobbit — J.R.R. Tolkien


SVD Recommendations:
--------------------
  [5448] Family Matters — Rohinton Mistry
  [9008] In Watermelon Sugar — Richard Brautigan
  [739] Different Seasons — Stephen King
  [6729] Ordinary People — Judith Guest
  [5865] Tales of Ordinary Madness — Charles Bukowski
  [5392] Regeneration (Regeneration, #1) — Pat Barker
  [2397] The Further Adventures of Sherlock Holmes: After Sir Arthur Conan Doyle (Classic Crime) — Richard Lancelyn Green, Ronald Knox, Julian Symons, Various
  [8316] The Inimitable Jeeves (Jeeves, #2) — P.G. Wodehouse
  [2612] A Suitable Boy (A Suitable Boy, #1) — Vikram Seth
  [8768] The Annotated Sherlock Holmes: The Four Novels and the Fifty-Six Short Stories Complete (2 Volume Set) — Arthur Conan Doyle, William S. Baring-Gould


Content-Based (TF-IDF) Recommendations:
---------------------------------------
  [4928] A Trick of the Light (Chief Inspector Armand Gamache, #7) — Louise Penny
  [4295] The Cruelest Month (Chief Inspector Armand Gamache, #3) — Louise Penny
  [4595] The Brutal Telling (Chief Inspector Armand Gamache, #5) — Louise Penny
  [3621] A Fatal Grace (Chief Inspector Armand Gamache, #2) — Louise Penny
  [2848] A God in Ruins — Kate Atkinson
  [4552] The Beautiful Mystery (Chief Inspector Armand Gamache, #8) — Louise Penny
  [5384] Hark! A Vagrant — Kate Beaton
  [4150] How the Light Gets In (Chief Inspector Armand Gamache, #9) — Louise Penny
  [5422] A Great Reckoning (Chief Inspector Armand Gamache, #12) — Louise Penny
  [5319] The Long Way Home (Chief Inspector Armand Gamache, #10) — Louise Penny


RL-Only Recommendations:
------------------------
  [3159] Betty Crocker's Cookbook — Betty Crocker
  [7860] The Way to Cook — Julia Child
  [2163] What to Expect When You're Expecting — Heidi Murkoff, Arlene Eisenberg, Sandee Hathaway
  [6902] Standing for Something: 10 Neglected Virtues That Will Heal Our Hearts and Homes — Gordon B. Hinckley
  [8533] The Cake Bible — Rose Levy Beranbaum, Maria Guarnaschelli, Vincent Lee, Manuela Paul, Dean G. Bornstein
  [5163] On Death and Dying — Elisabeth Kübler-Ross
  [7946] The Beauty Myth — Naomi Wolf
  [1058] Murder at the Vicarage (Miss Marple, #1) — Agatha Christie
  [8246] The Fannie Farmer Cookbook: Anniversary — Marion Cunningham, Fannie Merritt Farmer, Archibald Candy Corporation
  [1330] Don't Sweat the Small Stuff ... and it's all small stuff: Simple Ways to Keep the Little Things from Taking Over Your Life — Richard Carlson


Hybrid (CF + RL, Z-score) Recommendations:
------------------------------------------
  [4] To Kill a Mockingbird — Harper Lee
  [11] The Kite Runner — Khaled Hosseini
  [9] Angels & Demons  (Robert Langdon, #1) — Dan Brown
  [46] Water for Elephants — Sara Gruen
  [33] Memoirs of a Geisha — Arthur Golden
  [31] The Help — Kathryn Stockett
  [22] The Lovely Bones — Alice Sebold
  [38] The Time Traveler's Wife — Audrey Niffenegger
  [5] The Great Gatsby — F. Scott Fitzgerald
  [14] Animal Farm — George Orwell


==============================
📚 BOOK RECOMMENDER SYSTEM
==============================
1. Random user → compare all models
2. Enter a book title → compare Item-CF / SVD / RL / Hybrid / CBF
3. Enter a user ID → compare all models
4. Exit

Choose an option: 2

Enter a book title: A Crown of Swords

Closest matches:
----------------
1. A Crown of Swords (Wheel of Time, #7) (book_id=1119, score=90.0%)
2. Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) (book_id=18, score=85.5%)
3. The Fellowship of the Ring (The Lord of the Rings, #1) (book_id=19, score=85.5%)

Select 1/2/3 (Enter = 1): 1

You selected:
  [1119] A Crown of Swords (Wheel of Time, #7) — Robert Jordan

Item-CF Similar Books:
----------------------
  [1249] The Path of Daggers (Wheel of Time, #8) — Robert Jordan
  [1023] Lord of Chaos (Wheel of Time, #6) — Robert Jordan
  [949] The Fires of Heaven (Wheel of Time, #5) — Robert Jordan
  [1362] Winter's Heart (Wheel of Time, #9) — Robert Jordan
  [1525] Crossroads of Twilight (Wheel of Time, #10) — Robert Jordan
  [722] The Shadow Rising (Wheel of Time, #4) — Robert Jordan
  [1278] Knife of Dreams (Wheel of Time, #11) — Robert Jordan
  [528] The Dragon Reborn (Wheel of Time, #3) — Robert Jordan
  [510] The Great Hunt (Wheel of Time, #2) — Robert Jordan
  [960] The Gathering Storm (Wheel of Time, #12) — Robert Jordan, Brandon Sanderson


SVD-Embedding Similar Books:
----------------------------
  [1362] Winter's Heart (Wheel of Time, #9) — Robert Jordan
  [1023] Lord of Chaos (Wheel of Time, #6) — Robert Jordan
  [949] The Fires of Heaven (Wheel of Time, #5) — Robert Jordan
  [1249] The Path of Daggers (Wheel of Time, #8) — Robert Jordan
  [1278] Knife of Dreams (Wheel of Time, #11) — Robert Jordan
  [1525] Crossroads of Twilight (Wheel of Time, #10) — Robert Jordan
  [722] The Shadow Rising (Wheel of Time, #4) — Robert Jordan
  [528] The Dragon Reborn (Wheel of Time, #3) — Robert Jordan
  [510] The Great Hunt (Wheel of Time, #2) — Robert Jordan
  [960] The Gathering Storm (Wheel of Time, #12) — Robert Jordan, Brandon Sanderson


RL-Based Similar Books (avg user state):
----------------------------------------
  [255] Atlas Shrugged — Ayn Rand, Leonard Peikoff
  [287] The Fountainhead — Ayn Rand, Leonard Peikoff
  [903] Anthem — Ayn Rand
  [122] Wicked: The Life and Times of the Wicked Witch of the West (The Wicked Years, #1) — Gregory Maguire, Douglas Smith
  [505] Left Behind (Left Behind, #1) — Tim LaHaye, Jerry B. Jenkins
  [1039] Pride and Prejudice and Zombies (Pride and Prejudice and Zombies, #1) — Seth Grahame-Smith, Jane Austen
  [7384] Mister Pip — Lloyd Jones
  [992] The Twilight Saga (Twilight, #1-4) — Stephenie Meyer, Ilyana Kadushin, Matt Walters
  [1476] The Slippery Slope (A Series of Unfortunate Events, #10) — Lemony Snicket, Brett Helquist
  [1033] The Wide Window (A Series of Unfortunate Events, #3) — Lemony Snicket, Brett Helquist


Hybrid (Item-CF + RL) Similar Books:
------------------------------------
  [1249] The Path of Daggers (Wheel of Time, #8) — Robert Jordan
  [1023] Lord of Chaos (Wheel of Time, #6) — Robert Jordan
  [949] The Fires of Heaven (Wheel of Time, #5) — Robert Jordan
  [1362] Winter's Heart (Wheel of Time, #9) — Robert Jordan
  [1525] Crossroads of Twilight (Wheel of Time, #10) — Robert Jordan
  [722] The Shadow Rising (Wheel of Time, #4) — Robert Jordan
  [1278] Knife of Dreams (Wheel of Time, #11) — Robert Jordan
  [528] The Dragon Reborn (Wheel of Time, #3) — Robert Jordan
  [510] The Great Hunt (Wheel of Time, #2) — Robert Jordan
  [960] The Gathering Storm (Wheel of Time, #12) — Robert Jordan, Brandon Sanderson


Content-Based (TF-IDF) Similar Books:
-------------------------------------
  [949] The Fires of Heaven (Wheel of Time, #5) — Robert Jordan
  [1249] The Path of Daggers (Wheel of Time, #8) — Robert Jordan
  [9343] The Wheel of Time: Boxed Set  (Wheel of Time, #1-8) — Robert Jordan
  [6678] The Wheel of Time: Boxed Set #1 (Wheel of Time, #1-3) — Robert Jordan
  [330] The Eye of the World (Wheel of Time, #1) — Robert Jordan
  [1525] Crossroads of Twilight (Wheel of Time, #10) — Robert Jordan
  [722] The Shadow Rising (Wheel of Time, #4) — Robert Jordan
  [1362] Winter's Heart (Wheel of Time, #9) — Robert Jordan
  [510] The Great Hunt (Wheel of Time, #2) — Robert Jordan
  [1023] Lord of Chaos (Wheel of Time, #6) — Robert Jordan


==============================
📚 BOOK RECOMMENDER SYSTEM
==============================
1. Random user → compare all models
2. Enter a book title → compare Item-CF / SVD / RL / Hybrid / CBF
3. Enter a user ID → compare all models
4. Exit

Choose an option: 3

Enter user_id: 1245

🔎 Recommendations for user_id = 1245

User-CF Recommendations:
------------------------
  [3725] Rise of Empire (The Riyria Revelations, #3-4) — Michael J. Sullivan
  [523] The Things They Carried — Tim O'Brien
  [2889] Mistborn Trilogy Boxed Set (Mistborn, #1-3) — Brandon Sanderson
  [9141] The Way of Kings, Part 1 (The Stormlight Archive #1.1) — Brandon Sanderson
  [6218] From the Two Rivers: The Eye of the World, Part 1 (Wheel of time, #1-1) — Robert Jordan
  [2196] Foundation's Edge (Foundation #4) — Isaac Asimov
  [46] Water for Elephants — Sara Gruen
  [3798] Heir of Novron (The Riyria Revelations, #5-6) — Michael J. Sullivan
  [3474] Tower Lord (Raven's Shadow, #2) — Anthony  Ryan
  [4889] A Perfect Blood (The Hollows, #10) — Kim Harrison


Item-CF Recommendations:
------------------------
  [4867] Skin Trade (Anita Blake, Vampire Hunter #17) — Laurell K. Hamilton
  [5410] Flirt (Anita Blake, Vampire Hunter #18) — Laurell K. Hamilton
  [5370] Bullet (Anita Blake, Vampire Hunter #19) — Laurell K. Hamilton
  [5684] A Lick of Frost (Merry Gentry, #6) — Laurell K. Hamilton, Laural Merlington
  [6080] Swallowing Darkness (Merry Gentry, #7) — Laurell K. Hamilton
  [6143] Hit List (Anita Blake, Vampire Hunter #20) — Laurell K. Hamilton
  [21] Harry Potter and the Order of the Phoenix (Harry Potter, #5) — J.K. Rowling, Mary GrandPré
  [24] Harry Potter and the Goblet of Fire (Harry Potter, #4) — J.K. Rowling, Mary GrandPré
  [18] Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) — J.K. Rowling, Mary GrandPré, Rufus Beck
  [23] Harry Potter and the Chamber of Secrets (Harry Potter, #2) — J.K. Rowling, Mary GrandPré


SVD Recommendations:
--------------------
  [192] The Name of the Wind (The Kingkiller Chronicle, #1) — Patrick Rothfuss
  [1200] The Alloy of Law (Mistborn, #4) — Brandon Sanderson
  [746] The Lies of Locke Lamora (Gentleman Bastard, #1) — Scott Lynch
  [1602] Changes (The Dresden Files, #12) — Jim Butcher
  [307] The Wise Man's Fear (The Kingkiller Chronicle, #2) — Patrick Rothfuss
  [747] Sabriel (Abhorsen,  #1) — Garth Nix
  [1394] White Night (The Dresden Files, #9) — Jim Butcher
  [1450] Small Favor (The Dresden Files, #10) — Jim Butcher
  [1665] Warbreaker (Warbreaker, #1) — Brandon Sanderson
  [1654] Cold Days (The Dresden Files, #14) — Jim Butcher


Content-Based (TF-IDF) Recommendations:
---------------------------------------
  [3341] The Bands of Mourning (Mistborn, #6) — Brandon Sanderson
  [1049] Elantris (Elantris, #1) — Brandon Sanderson
  [1665] Warbreaker (Warbreaker, #1) — Brandon Sanderson
  [1200] The Alloy of Law (Mistborn, #4) — Brandon Sanderson
  [7993] Secret History (Mistborn, #3.5) — Brandon Sanderson
  [9141] The Way of Kings, Part 1 (The Stormlight Archive #1.1) — Brandon Sanderson
  [2792] Shadows of Self (Mistborn, #5) — Brandon Sanderson
  [970] Steelheart (The Reckoners, #1) — Brandon Sanderson
  [2118] Firefight (The Reckoners, #2) — Brandon Sanderson
  [3249] Calamity (The Reckoners, #3) — Brandon Sanderson


RL-Only Recommendations:
------------------------
  [2807] The Good, the Bad, and the Undead (The Hollows, #2) — Kim Harrison
  [6534] Grave Secret (Harper Connelly, #4) — Charlaine Harris
  [1273] Dead Witch Walking (The Hollows, #1) — Kim Harrison
  [9488] Festive in Death (In Death, #39) — J.D. Robb
  [1228] Death Masks (The Dresden Files, #5) — Jim Butcher
  [1034] Grave Peril (The Dresden Files, #3) — Jim Butcher
  [4997] Fair Game (Alpha & Omega, #3) — Patricia Briggs
  [1546] The Black Prism (Lightbringer, #1) — Brent Weeks
  [2430] Grave Sight (Harper Connelly, #1) — Charlaine Harris
  [5044] Grave Surprise (Harper Connelly, #2) — Charlaine Harris


Hybrid (CF + RL, Z-score) Recommendations:
------------------------------------------
  [4867] Skin Trade (Anita Blake, Vampire Hunter #17) — Laurell K. Hamilton
  [5410] Flirt (Anita Blake, Vampire Hunter #18) — Laurell K. Hamilton
  [5370] Bullet (Anita Blake, Vampire Hunter #19) — Laurell K. Hamilton
  [5684] A Lick of Frost (Merry Gentry, #6) — Laurell K. Hamilton, Laural Merlington
  [6143] Hit List (Anita Blake, Vampire Hunter #20) — Laurell K. Hamilton
  [8822] Concealed in Death (In Death, #38) — J.D. Robb
  [6080] Swallowing Darkness (Merry Gentry, #7) — Laurell K. Hamilton
  [25] Harry Potter and the Deathly Hallows (Harry Potter, #7) — J.K. Rowling, Mary GrandPré
  [27] Harry Potter and the Half-Blood Prince (Harry Potter, #6) — J.K. Rowling, Mary GrandPré
  [18] Harry Potter and the Prisoner of Azkaban (Harry Potter, #3) — J.K. Rowling, Mary GrandPré, Rufus Beck


==============================
📚 BOOK RECOMMENDER SYSTEM
==============================
1. Random user → compare all models
2. Enter a book title → compare Item-CF / SVD / RL / Hybrid / CBF
3. Enter a user ID → compare all models
4. Exit

Choose an option: 4
```
