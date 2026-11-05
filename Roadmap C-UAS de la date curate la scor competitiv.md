# Roadmap C-UAS: de la date curate la scor competitiv

Oct 28, 2026 · @Teodor

**Pe scurt:** construiești întâi măsurarea, apoi tracking-ul și fuziunea, care dau 70% din scor, și abia la final ML-ul. Totul trebuie să ruleze cu o singură comandă pe date noi, pentru că primele cinci echipe își rulează soluția live în fața juriului. Ghidul e aliniat la cele patru documente oficiale NCIA: instrucțiunile și cele trei README-uri.

## 1. Regulile și scorul decid strategia

70% din punctaj (GOSPA 0,4 + Fusion Completeness 0,3) vine din tracking și fuziune, nu din modele de ML. Modelele clasice mută direct cel mult 0,2–0,3. Deci „multe modele cu performanțe excepționale” e etapa finală, nu prima.

### Cum se câștigă, după regulile oficiale

Leaderboard-ul te duce doar în top 5; clasamentul final se decide după un demo live, pe date pe care nu le-ai văzut.

1. **Faza 1, antrenare.** Îți construiești soluția pe setul de Testing, singurul cu ground truth.
2. **Faza 2, evaluare.** Încarci câte un .xlsx pentru fiecare set de evaluare. Poți încărca de mai multe ori, iar după închidere se păstrează, pe fiecare set, cel mai bun scor. Primele cinci echipe trec mai departe.
3. **Faza 3, juriu.** O oră online: 45 de minute de prezentare și demo live pe un set nou, primit pe loc, plus 15 minute de întrebări. Top 5 trimite un PDF de metodologie; top 3, dovezi sau un prototip.

Ce înseamnă asta pentru tine:

- **O singură comandă, pe date noi, fără ajustări manuale.** Orice reglaj făcut de mână pentru un scenariu anume cade la demo.
- **Timpul de rulare contează.** Demo-ul încape în cele 45 de minute, împreună cu prezentarea. Aici un nucleu C++ pentru asociere și filtrare chiar aduce ceva.
- **ML-ul contează și pentru juriu.** Regulamentul cere o soluție „AI-enabled”; scorul de asociere învățat și clasificatorul sunt partea pe care o prezinți ca atare.
- **Premiul cere un sponsor.** Fără sponsorizarea unei universități, a unui institut de cercetare sau a unei firme apari doar în clasament. Fondul e de 100.000 €: 45.000, 35.000 și 20.000 € pentru primele trei locuri.

Data de închidere nu apare în instrucțiuni, iar ceremonia e plasată o dată în noiembrie 2026 și o dată în februarie 2027. Pagina NATO a evenimentului indică 21 decembrie 2026; o confirmi prin email.

### Ce date ai și ce îți spun

| Scenariu | Testing, cu ground truth | Evaluare, fără ground truth |
| --- | --- | --- |
| 1 | iterația 1 | iterația 3; iterația 2, listată separat ca „Phase 2” |
| 2 | iterația 2 | iterațiile 1 și 3 |
| 3 | iterațiile 1, 2 și 3 | — |
| 4 | iterația 4 | — |

- **Leaderboard-ul e pe scenariile 1 și 2.** S1 I1 și S2 I2 din Testing sunt cele mai apropiate de el. Scenariile 3 și 4 îți arată cum te descurci pe un scenariu nou, adică exact situația demo-ului.
- **Etichetele vin din DroneTag.** Echipa din numele fișierului (BLUE sau RED) dă identitatea, iar numele dronei dă modelul și clasa. Serialul, protocolul și banda nu apar în ground truth, deci Fusion Completeness se poate măsura local doar parțial.
- **Datele de evaluare servesc doar la submisii.** Nu antrenezi pe ele și nu faci pseudo-labeling pe ele.
- **Numără subfolderele.** Textul spune că Testing are opt, tabelul arată șase iterații.

### Cum se calculează scorul

| Componentă | Pondere | Ce o mișcă | Unde ajută ML-ul |
| --- | --- | --- | --- |
| Tracking (GOSPA) | 0,4 | duplicate eliminate, falsuri eliminate, poziție precisă | scor de asociere învățat, filtru „e dronă reală?” |
| Fusion Completeness | 0,3 | cele 4 coloane de metadate (model, protocol, serial, bandă) corecte, pe track-ul corect | puțin: e asociere și parsare |
| Clasificare (recall) | 0,2 | Fixed Wing UAS, Rotary Wing UAS sau UAS Transmitter; ce nu e dronă nu se raportează | aici: GBDT pe features de track |
| Identificare (recall) | 0,1 | Friendly sau Hostile, din serialul comparat cu lista de drone prietene | aproape deloc: reguli |

**De ce nu e un concurs Kaggle obișnuit.** La Kaggle tabular, rândurile de clasificat există deja în fișier. Aici rândurile, adică track-urile fuzionate, nu există până nu le construiești tu. Un clasificator perfect peste track-uri duplicate pierde tot la GOSPA.

**Cum gândește GOSPA.** La fiecare pas de 100 ms, metrica împerechează optim track-urile tale cu dronele reale. O pereche costă distanța ridicată la puterea p, plafonată la c. Orice dronă ratată și orice track fals costă c^p/2.

```latex
d_p^{(c,2)}(X,Y) = \Big( \min_{\gamma} \sum_{(i,j)\in\gamma} d^{(c)}(x_i,y_j)^p + \frac{c^p}{2}\big(|X| + |Y| - 2|\gamma|\big) \Big)^{1/p}
```

**Consecința practică: raportezi un track doar dacă probabilitatea P să fie o dronă reală trece de 50%.** Raportat, costul așteptat e (1 − P)·c^p/2 + P·d^p. Neraportat, e P·c^p/2. Pentru d mic, raportarea câștigă doar când P > 0,5; cu o poziție imprecisă, pragul urcă.

Documentele oficiale spun doar „metoda GOSPA”: valorile c și p, distanța 2D sau 3D și o eventuală penalizare la schimbarea de ID rămân întrebări pentru organizatori (secțiunea 8). Covarianța nu intră în GOSPA-ul standard, așa că o completezi corect, dar nu investești în ea până nu afli că e notată.

## 2. K-means: ce păstrezi și ce înlocuiești

K-means e bun ca explorare, dar nu poate fi baza deduplicării. Ignoră timpul, cere numărul de obiecte dinainte și nu știe că un senzor nu vede același obiect de două ori în același moment.

**De ce cedează pe datele astea:**

- **Cere K.** Numărul de drone variază în timp, și chiar el e o parte din ce estimezi.
- **Presupune nori rotunzi.** Un radar măsoară bine distanța și mai slab unghiul, deci eroarea lui e o elipsă alungită. Distanța Mahalanobis, cu covarianța R a senzorului, ține cont de forma asta; distanța euclidiană nu.
- **Nu vede timpul.** Două drone care se încrucișează se contopesc într-un cluster; o dronă rapidă se rupe în mai mulți.
- **Nu respectă constrângeri.** Poate pune două detecții simultane ale aceluiași senzor în același cluster, deși sunt sigur obiecte diferite.
- **Nu are loc pentru zgomot.** Fiecare punct e forțat într-un cluster, inclusiv păsările și clutter-ul radar.
- **Nu toți senzorii dau un punct 3D.** OEPS și GTYT (RF) dau doar azimut, SMFS doar latitudine și longitudine, iar WCFX, ORFH și EMXGA dau distanță și azimut, fără elevație.

**Ce păstrezi:** rezultatele, ca diagnostic: câte obiecte par să existe per scenariu, ce senzor pare decalat față de ceilalți. Întrebare deschisă: ce ai clusterizat exact (detecții sau track-uri, pe ce fereastră de timp)?

**Ce îl înlocuiește:**

| Problema | Unealta | De ce funcționează |
| --- | --- | --- |
| Ce măsurătoare poate aparține cărui track | Gating Mahalanobis, cu prag χ² (de exemplu 99%) | Folosește forma reală a erorii fiecărui senzor |
| Cine cu cine, unu-la-unu, la un moment dat | Algoritmul Hungarian (GNN), `scipy.optimize.linear_sum_assignment` | Optim global; garantează că o detecție merge la un singur track |
| Track-uri de la senzori diferiți = același obiect | Asociere track-to-track pe ferestre de timp, apoi unire greedy în ordinea scorului, cu cannot-link | Compară istorii întregi, nu puncte izolate; refuză orice unire care pune doi track-uri simultane ale aceluiași senzor în același obiect |
| Varianta cu ML | Clasificator pe perechi de track-uri, etichetat din ground truth | Învață din date ce perechi sunt același obiect (secțiunea 5) |

De ce unire greedy, nu simple componente conexe: componentele conexe înlănțuie A–B și B–C chiar dacă A și C sunt sigur obiecte diferite. Unirea în ordinea scorului, cu verificarea constrângerii la fiecare pas, oprește lanțul acolo.

### Cum intră fiecare senzor în fuziune

Cei 20 de senzori din Sensor README dau cinci tipuri de măsurătoare, iar fiecare tip are propriul model de măsurătoare în filtru.

| Ce măsoară | Senzori | Cum intră în filtru |
| --- | --- | --- |
| Poziție 3D (lat, lon, alt) | radare IUKL, UXYJ, EGOI, NYTF, EUIG; RF ILYS, MMJJ, IKJX; cyber CEQX, STUG; UTM VKYY, QTVS | actualizare liniară de poziție, după conversia în ENU |
| Poziție 2D (lat, lon) | RF SMFS | actualizare doar pe Est și Nord; altitudinea vine din alte surse |
| Distanță, azimut, elevație | EO/IR NJQL, FGIX | model sferic, cu EKF sau UKF |
| Distanță și azimut | RF ORFH, EO/IR WCFX, acustic EMXGA | model polar în plan orizontal, fără informație de înălțime |
| Doar azimut | RF OEPS, GTYT | o dreaptă de direcție: confirmă și rafinează un track existent, nu pornește unul nou |

- **Vitezele.** UXYJ, EGOI, NYTF, EUIG, CEQX, VKYY și QTVS raportează și viteza în ENU; STUG doar pe Est și Nord. O folosești ca măsurătoare și ca feature pentru asociere.
- **Azimutul e față de „nordul propriu” al senzorului.** Fiecare senzor unghiular are un bias de orientare, pe care îl estimezi pe ground truth.
- **Verifică UTM-ul întâi.** VKYY și QTVS sunt sisteme de management al traficului de drone. Dacă poziția lor vine chiar de la dronele cooperante, pot fi cea mai precisă sursă pe care o ai; măsori asta pe ground truth înainte să le dai pondere.
- **Senzorii folosesc 17 etichete, ieșirea doar cinci.** „Bird”, „Manned Fixed Wing” sau „Home” spun că obiectul nu se raportează, iar „UAV Rotary Wing” indică „Rotary Wing UAS”. Cât să crezi fiecare senzor înveți pe Testing, din matricea lui de confuzie.

## 3. Metoda competitivă: o buclă, nu o listă de modele

Echipele de top nu au mai multe modele, ci mai multe experimente în care pot avea încredere. [Playbook-ul grandmasterilor Kaggle de la NVIDIA](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) pune la bază exact două lucruri: experimentare rapidă și validare atentă.

1. **Validare locală de încredere.** Scorer-ul tău (GOSPA + recall) rulat pe setul de Testing, care are ground truth. Lași deoparte câte o iterație întreagă (leave-one-iteration-out), nu rânduri. S1 I1 și S2 I2 îți aproximează leaderboard-ul, iar S3 și S4 îți arată ce se întâmplă pe un scenariu nou, ca la demo.
   - *De ce:* rândurile din același zbor sunt aproape identice. Un split aleator pune același zbor și în antrenare, și în validare, iar scorul iese umflat.
2. **Baseline complet, oricât de slab.** „Fiecare detecție radar = un track” și „cel mai bun senzor singur”, de la datele brute până la fișierul de submisie.
   - *De ce:* îți dă podeaua de scor și testează tot lanțul, inclusiv formatul.
3. **Prima submisie, devreme.**
   - *De ce:* verifici că scorul local și cel oficial se mișcă în același sens. Dacă nu, validarea locală e greșită și orice decizie ulterioară e zgomot. Pe fiecare set contează cea mai bună submisie, deci încărcările dese nu te costă nimic; aștepți doar ca fișierul să fie mutat în folderul datat, altfel îl suprascrii.
4. **Analiza erorilor.** Descompui GOSPA în localizare, ratări și falsuri, pe scenariu și pe senzor. Ataci cea mai mare bucată.
   - *De ce:* îți arată unde e următorul punct, în loc să ghicești.
5. **O singură schimbare per experiment, notată.** Un `experiments.csv` cu data, commit-ul, schimbarea și scorul pe componente.
   - *De ce:* dacă schimbi trei lucruri și scorul urcă, nu știi care a ajutat; unul poate chiar să fi stricat.
6. **Repeți pașii 4 și 5** până când câștigurile devin mici.
7. **Abia apoi: tuning și ansamblu.** Mai multe seed-uri, apoi combinarea modelelor pe predicțiile out-of-fold.
   - *De ce la final:* un ansamblu peste un pipeline greșit doar face greșeala mai stabilă.
8. **Îngheț cu câteva zile înainte de termen.** Rerulezi totul de la zero, cu o singură comandă, pe o mașină curată, și validezi formatul. Exact așa va arăta demo-ul live.

## 4. Roadmap până pe 21 decembrie

Ai aproape opt săptămâni, iar ordinea contează mai mult decât viteza: măsurare, tracking, fuziune, atribute, abia apoi ML. Fiecare etapă se închide cu o poartă, adică un test concret pe care îl treci înainte să mergi mai departe.

&#91;embedded content: roadmap · 6 etape, fiecare închisă de o poartă\]

Prima submisie pleacă în săptămâna 2, ca să afli devreme dacă scorul local seamănă cu cel oficial. Instrucțiunile oficiale nu dau data de închidere, iar pagina NATO indică 21 decembrie; o confirmi prin email (secțiunea 8). Dacă intri în top 5, urmează sesiunea cu juriul, la o dată încă neanunțată.

## 5. Unde intră ML-ul și ce modele merită

ML-ul are trei locuri bune aici, toate pe track-uri, nu pe rânduri brute: scorul de asociere, filtrul „e dronă reală?” și clasificatorul de clasă. Primele două mută GOSPA; al treilea mută recall-ul de clasificare.

| Model | Ce clasifică | Eticheta, din ground truth | Ce parte din scor mută |
| --- | --- | --- | --- |
| Scor de asociere | o pereche de track-uri de la senzori diferiți | sunt același obiect real? | GOSPA: duplicatele |
| Filtru de existență | un track fuzionat | corespunde unei drone reale? | GOSPA: falsurile |
| Clasificator de clasă | un track fuzionat | Fixed Wing UAS, Rotary Wing UAS sau UAS Transmitter, din numele dronei | Clasificare (0,2) |

**Etichetele.** Clasa și modelul le derivi din numele dronei din fișierul DroneTag, cu o mapare scrisă de tine către valorile din dropdown; identitatea vine din BLUE sau RED. Ieșirea nu are clasă pentru păsări sau aeronave pilotate, deci filtrul de existență face și separarea „dronă vs. pasăre”: ce nu e dronă pur și simplu nu se raportează.

**Features pe track.** Fiecare e un rezumat al istoriei track-ului:

- **Cinematică:** viteză medie și maximă, accelerație, rata de viraj, altitudinea și variația ei, cât timp stă pe loc. *De ce:* dronele multirotor pot staționa și urca vertical, pe când majoritatea păsărilor nu.
- **Acoperire:** câți senzori l-au văzut, cât a durat, câte actualizări are. *De ce:* un fals apare de obicei scurt și la un singur senzor.
- **Ce spun senzorii:** eticheta raportată de fiecare senzor, detectionConfidence, behaviour (în mișcare sau staționar), lista de semnale RF și prezența unui raport cyber. *De ce:* modelul învață cât să creadă fiecare senzor, adică ponderile pe care altfel le-ai pune de mână.
- **Pentru perechi:** distanța Mahalanobis medie pe suprapunere, diferența de viteză, reziduul de azimut, durata suprapunerii, tipul perechii de senzori.

**Modelul: GBDT întâi** (LightGBM, XGBoost sau CatBoost). *De ce:* lucrează bine cu features eterogene, tolerează valorile lipsă (senzorul care n-a văzut obiectul) și merge pe puține exemple. Rețelele neuronale cer mult mai multe track-uri etichetate decât oferă câteva scenarii.

**Calibrarea probabilităților.** Regula de raportare din secțiunea 1 presupune că „0,7” chiar înseamnă 70%. GBDT nu garantează asta, deci calibrezi pe predicțiile out-of-fold (regresie izotonică sau Platt). *De ce:* altfel pragul de 50% e aplicat pe o scară greșită.

**„Unknown” și „Not Applicable”.** Template-ul folosește „Unknown” când algoritmul nu poate estima și „Not Applicable” când lipsesc datele; cum le numără recall-ul e întrebare pentru organizatori. Pentru identitate, la o dronă fără serial potrivit, alegi între Unknown și Hostile pe date: măsori pe Testing ce dă recall mai mare. Păstrezi probabilitățile complete și decizi eticheta abia la final.

**Ce merită la ansamblu:**

- Media pe mai multe seed-uri și pe 2–3 biblioteci GBDT. Ieftin și aproape mereu ajută.
- Hill climbing: pornești de la cel mai bun model și adaugi altele cu ponderi, doar cât timp scorul out-of-fold crește.

**Ce nu merită aici:** stacking pe mai multe niveluri și rețele mari. Cu doar câteva iterații etichetate, ele învață particularitățile scenariilor, nu dronele. Pseudo-labeling pe datele de evaluare iese din discuție: regulile le rezervă doar pentru submisii.

## 6. Resurse disponibile imediat

Cea mai valoroasă resursă e precedentul direct: NCIA a mai rulat o provocare C-UAS pe Kaggle în 2021, iar [organizatorii au recomandat notebook-ul Dstl](https://www.ncia.nato.int/about-us/newsroom/agency-announces-winners-of-drone-data-challenge) ca exemplu de rezolvare. Dstl a fuzionat acolo două radare și două goniometre RF cu Stone Soup.

| Resursă | Tip, cost | Ce iei din ea |
| --- | --- | --- |
| [ICMCIS drone tracking, Kaggle 2021](https://www.kaggle.com/c/icmcis-drone-tracking/) | competiție, gratuit | Aceeași problemă pe date C-UAS reale; caută notebook-ul Dstl în tab-ul Code |
| [Slide-urile Dstl despre Stone Soup (J. Barr, 2022)](https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/attachments/Jordi%20Barr.pdf) | PDF, gratuit | Rețeta Dstl: EKF cu viteză aproape constantă, Mahalanobis + GNN, track-uri pornite doar din radar, ștergere după timp și incertitudine |
| [Tutorialele Stone Soup](https://stonesoup.readthedocs.io/en/latest/auto_tutorials/index.html) | documentație, gratuit | Kalman, asociere, inițiere și ștergere de track-uri, tracker complet |
| [GOSPAMetric din Stone Soup](https://stonesoup.readthedocs.io/en/v1.9/stonesoup.metricgenerator.ospametric.html) | cod, gratuit | Implementare de referință, ca să-ți verifici scorer-ul; dă separat localizarea, ratările și falsurile |
| [Articolul GOSPA (Rahmathullah, García-Fernández, Svensson)](https://arxiv.org/abs/1601.05585) | articol, gratuit | Exact metrica după care ești notat |
| [Kalman and Bayesian Filters in Python (R. Labbe)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) | carte în notebook-uri, gratuit | Kalman, EKF, UKF și smoothing, cu cod de rulat |
| [Cyrill Stachniss: Kalman Filter & EKF](https://www.ipb.uni-bonn.de/online-training-robotics/) | video, gratuit | Teoria filtrului explicată pas cu pas; există și varianta de 5 minute |
| [Approaching (Almost) Any ML Problem (A. Thakur)](https://github.com/abhishekkrthakur/approachingalmost) | carte PDF, gratuit | Cross-validation, metrici, feature engineering, ensembling și stacking |
| [Kaggle Grandmasters Playbook (NVIDIA, 2025)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) | articol, gratuit | Validare, baseline-uri diverse, hill climbing, media pe seed-uri |
| The Kaggle Book, ed. a 2-a (Banachewicz, Massaron, Tunguz) | carte, plătită | Capitolele despre validare și ensembling; opțională |

**Ordinea pentru primele trei seri:**

1. Slide-urile Dstl, apoi notebook-ul de pe Kaggle: vezi o soluție completă la aceeași problemă.
2. Tutorialele Stone Soup 1, 5, 6, 9 și 10: Kalman, asociere, ciclul de viață al track-urilor, totul legat.
3. Thakur, capitolele despre cross-validation și ensembling, apoi Playbook-ul NVIDIA.

## 7. Capcane și fișierul de submisie

Cele mai multe concursuri se pierd din validare și format, nu din model.

- **Split pe rânduri în loc de iterații.** Scorul local iese umflat, iar pe evaluare se prăbușește. Validezi doar cu iterații lăsate deoparte întregi.
- **Reglaje diferite pe fiecare set de evaluare.** Regula „cea mai bună submisie pe set” te tentează să reglezi separat fiecare set. Juriul rulează însă soluția live pe date noi, deci contează doar ce face aceeași configurație peste tot. Decizi pe scorul local; leaderboard-ul confirmă direcția.
- **Raportezi tot ce vezi.** Păsările și clutter-ul raportate ca track-uri costă cât dronele ratate. Aplici regula de peste 50% din secțiunea 1.
- **Unități și repere amestecate.** README-ul spune grade unde datele sunt în radiani, și invers; ai prins deja asta. CEQX, de exemplu, dă erorile de poziție în grade, nu în metri. Verifici fiecare senzor pe ground truth, nu pe documentație.
- **Timpul.** Totul în aceeași referință (ns UTC), cu latența fiecărui senzor estimată. O latență de 0,5 s la 15 m/s înseamnă 7,5 m eroare de poziție.
- **Uiți decriptarea înainte de upload.** Lucrezi pe coordonate anonimizate; fișierul trimis trebuie readus în coordonatele reale și verificat.
- **Formatul.** Șiruri exacte din dropdown-uri, grila de 100 ms, numărul corect de rânduri pe timestamp. Un validator de format rulează înaintea fiecărui upload.
- **Mai multe schimbări odată.** Nu mai știi ce a ajutat și ce a stricat.
- **Procesare offline neclarificată.** Smoothing-ul înainte-înapoi poate fi permis sau nu. Afli înainte să construiești pe el.
- **Totul în ultima săptămână.** Rerularea completă de la zero se face cu cel puțin patru zile înainte de termen.

### Fișierul de submisie, câmp cu câmp

Fiecare rând de mai jos vine din Submission Template README; ultima coloană arată unde se pierd de obicei puncte.

| Câmp | Ce cere README-ul | Atenție la |
| --- | --- | --- |
| Fișierul | câte un `scenario_<N>_iteration_<M>.xlsx` pe set, cu numele original | scrii pe template-ul primit, nu într-un fișier nou |
| `timestamp`, `unix_timestamp` | grilă de 100 ms, de la primul la ultimul timestamp din ground truth; nu se modifică | filtrul prezice starea exact pe grilă; în afara ei nu raportezi nimic |
| Rândurile | un rând per țintă; K rânduri pentru K ținte; rând gol când nu ai track-uri | numărul de rânduri pe timestamp e chiar cardinalitatea notată de GOSPA |
| `latitude`, `longitude`, `altitude_m` | grade WGS 84; altitudinea peste elipsoidul WGS 84 | în DroneTag asta e `geo_altitude`, nu `altitude` (barometric) sau `height` (de la decolare) |
| `C_nn` … `C_uu` | triunghiul superior al covarianței, în m², pe axele Nord-Est-Sus | ordinea e NEU, nu ENU (nota de mai jos) |
| `track_id` | un UUID per obiect, același pe toată iterația | când un track se rupe și revine, păstrezi același UUID |
| `classification` și încrederea | Fixed Wing UAS, Rotary Wing UAS, UAS Transmitter, Unknown, Not Applicable; încredere între 0 și 1 | nu există clasă pentru păsări |
| `identity` | Hostile, Friendly, Unknown, Not Applicable | Friendly doar cu potrivire pe lista de drone prietene |
| Cele 4 coloane de metadate | valori fixe din dropdown (Tabelul C.4) | le citești programatic din template: au greșeli de tipar și liniuțe lungi, ca în „2400 – 2483.5”, care se copiază exact |

- **Covarianța, pe calea scurtă.** Dacă filtrul lucrează în metri, în ENU local, covarianța e deja carteziană. La 5 km de origine, planul tangent al dronei e rotit cu circa 0,05° față de al tău, deci ajunge să reordonezi ENU în NEU. Formula cu Jacobianul (C.3) e pentru cine ține starea în latitudine și longitudine. În textul README-ului, matricea ECEF→NEU apare transpusă față de forma standard, în care rândurile sunt vectorii Nord, Est și Sus. Un test tranșează: un punct mutat 1 m spre nord trebuie să dea n = 1, e = 0, u = 0.
- **Altitudinea.** Offsetul vertical al fiecărui senzor față de `geo_altitude` îl estimezi pe ground truth. Nu aplici un model de geoid pe coordonate anonimizate: după rotația în longitudine, ar fi calculat în alt loc de pe glob.
- **„Not Applicable”.** Regulile scriu „Not applicable”, dropdown-urile „Not Applicable”; folosești varianta din dropdown.
- **Metadatele se propagă pe tot track-ul.** Un senzor cyber poate citi serialul o singură dată. Odată ce track-ul e legat de el, completezi aceeași valoare pe toate rândurile track-ului: înainte în timp mereu, înapoi doar dacă procesarea offline e permisă.
- **„Home” nu e un track.** Senzorii raportează și poziții „Home”, probabil ale pilotului sau ale punctului de decolare. Ground truth-ul conține doar drone, deci un astfel de rând ar fi un fals. Ce corespunde clasei „UAS Transmitter” în ground truth rămâne de întrebat.
- **Contradicție deschisă.** Secțiunea 2.4.1 spune că Fusion Completeness descrie senzorii folosiți pentru fiecare track, iar exemplul 1.2.7 are coloanele `node_id` și `object-id`, absente din template. Secțiunea 2.5 spune că se notează corectitudinea metadatelor. Păstrezi oricum proveniența, ca să poți completa coloanele dacă se confirmă.

## 8. Ce faci mâine

Două lucruri nu pot aștepta: sponsorizarea, de care depinde premiul, și scorer-ul local, fără de care nu știi dacă K-means sau orice altă idee te-a ajutat.

- [ ] Cere sponsorizarea facultății (FII, UAIC), printr-un profesor sau prin decanat: fără un sponsor apari doar în clasament, fără premiu. Regulamentul acceptă și întreprinzătorii individuali ca furnizori comerciali; dacă vrei varianta asta, confirm-o cu organizatorii.
- [ ] Trimite la data.challenge@ncia.nato.int întrebările deschise: data de închidere, forma sponsorizării, parametrii GOSPA (c, p, 2D sau 3D, penalizare la schimbarea de ID), cum se numără Unknown și Not Applicable, ce înseamnă UAS Transmitter în ground truth, dacă node\_id și object-id trebuie completate și dacă procesarea offline e permisă.
- [ ] Construiește etichetele: maparea nume dronă → model din dropdown și clasă, plus BLUE/RED → identitate.
- [ ] Scrie scorer-ul: GOSPA pe grila de 100 ms a template-ului, descompus în localizare, ratări și falsuri, plus recall pe clasă și pe identitate.
- [ ] Verifică scorer-ul pe 2–3 cazuri mici, calculate de mână și cu `GOSPAMetric` din Stone Soup.
- [ ] Scrie writer-ul și validatorul de submisie, cu valorile dropdown citite din template, și testează-le pe un fișier gol.
- [ ] Baseline 0: fiecare detecție radar devine un track. Scorează-l, rulat cu o singură comandă.
- [ ] Baseline 1: cel mai bun senzor singur, cu VKYY și QTVS incluși în comparație. Scorează-l.
- [ ] Pornește `experiments.csv` și notează ambele baseline-uri.
- [ ] Notează ce ai clusterizat cu K-means și ce ai observat; devine primul diagnostic.

## Surse

- Documentele oficiale NCIA, de pe extranet: Data Challenge Instructions, Submission Template README, Sensor README, Ground Truth README
- [NATO: pagina evenimentului C-UAS Data Challenge 2026](https://www.nato.int/en/work-with-us/business-and-project-opportunities/events/2026/12/nato-c-uas-data-challenge-2026): data de 21 decembrie 2026
- [ICE Bruxelles: anunțul cu termenele](https://www.ice.it/it/news/notizie-dal-mondo/309951): înregistrare până pe 14 decembrie, propuneri până pe 21 decembrie
- [NCIA: câștigătorii provocării din 2021](https://www.ncia.nato.int/about-us/newsroom/agency-announces-winners-of-drone-data-challenge)
- [Dstl, J. Barr: Stone Soup și provocarea C-UAS de pe Kaggle](https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/attachments/Jordi%20Barr.pdf)
- [NVIDIA: The Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Stone Soup: GOSPAMetric](https://stonesoup.readthedocs.io/en/v1.9/stonesoup.metricgenerator.ospametric.html)
