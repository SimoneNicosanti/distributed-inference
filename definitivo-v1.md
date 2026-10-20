# Supposizioni 
## Infrastruttura
In termini di infrastruttura abbiamo quanto segue:
- Tre livelli di calcolo:
	- Device
	- Edge (forse meglio Fog a livello di nomenclatura)
	- Cloud
- Caratteristiche
	- Edge e Cloud.
	  Questi livelli fanno parte dell'infrastruttura condivisa da tutti gli utenti
	- Device
	  Supponiamo che ogni device appartenga ad un utente specifico, quindi ogni device esegue la computazione del modello scelto dall'utente che lo possiede (i.e., il device di utente A esegue la computazione del modello del solo utente A, e non di B)
- Connettività
	- Supponiamo che i tre livelli di calcolo siano tutti collegati tra loro
		- In funzione della distanza e delle capacità delle schede di rete, ci saranno delle bande e delle latenze diverse
		- Questo significa che un dispositivo qualsiasi può comunicare con qualsiasi altro dispositivo nel sistema. Questo vale con l'eccezione dei dispositivi nel livello Device: visto che device A non fa calcoli per utente B, possiamo supporre che non ci sia connettività tra loro (i.e., un dispositivo nel livello di calcolo Device, vede se stesso come unico dispositivo all'interno di quel livello).
	- La comunicazione può avvenire in entrambe le direzioni

![[diagram_20260615095322.drawio.svg]]

## Flussi
Dato un flusso di richieste, questo flusso ha associati requisiti in termini di:
- Tempo medio di risposta
- Consumo medio di energia
- Accuratezza dell'inferenza

Dato un flusso abbiamo quanto segue:
- Il flusso genera richieste singole ad un certo tasso $\Lambda^f$ (supponiamo processo di Poisson)
- Ogni flusso ha un certo insieme di modelli tra cui può scegliere $\mathcal{M}^f$. 
	- Questi modelli possono differire per diversi aspetti, ad esempio:
		- Per uso o meno della quantizzazione
		- Uso di diverse dimensioni di batching
		- Dimensione della stessa famiglia di modelli
		- Accuratezze
	- Possiamo distinguere anche tra modelli fine-tuned e specifici per l'utente e modelli generici disponibili a livello di sistema

Per ogni flusso dovremo:
- Scegliere un modello tra quelli possibili per lui
- Dividere il modello scelto sui vari server
Facendo ciò in funzione dei requisiti del flusso.

## Modello della DNN
Consideriamo la DNN come un DAG $G = (V, E, T)$. Nello specifico:
- Per ogni nodo $v \in V$ consideriamo:
	- FLOPS
	- Dimensione dei tensori coinvolti nel modello
- Per ogni arco $e \in E$ consideriamo:
	- Nodo inzio
	- Nodo fine
	- Dimensione dei dati trasmessi
- Per ogni tensore $t \in T$ consideriamo:
	- Nodo sorgente del tensore
	- Insieme di nodi che ricevono il tensore
	- Dimensione del tensore
- Per il modello
	- Dimensione del batch
	- Caratteristiche (in senso lato) di quantizzazione e/o accuratezza

## Divisione della DNN
Supponiamo che il modello venga diviso come una **pipeline** di insieme di livelli.
![[diagram_20260619122238.drawio.svg|700]]


Un partizionamento come quello in figura sicuramente garantisce aciclicità, ma non è detto che garantisca linearità. Infatti, se consideriamo i gruppi dell'esempio in figura, otteniamo quanto segue.
![[diagram_20260623143924.drawio.svg]]

Questa di fatto non è una divisione lineare, perché il quadrato blu ha uno skip verso il quadrato verde.
Quello che possiamo fare è trasferire il contesto che è ancora necessario:
![[diagram_20260623145955.drawio.svg]]
Cioè il blocco arancione fa da proxy per il trasferimento del tensore A al blocco verde, oltre a trasferire il tensore C.

Questo può implicare il trasferimento del tensore A anche più di una volta: sarà la fase di ottimizzazione a valutare se e quanto questo trasferimento aggiuntivo valga la pena.

Questo significa trasportare il contesto di esecuzione del modello attraverso i vari blocchi (e quindi server a cui i blocchi sono assegnati) fino a quando non sono più necessari.
## Modello del singolo server
Per un server possiamo considerare tre risorse principali:
- Capacità di calcolo
- Capacità di trasmissione
- Memoria
	- RAM nel caso di dispositivi senza GPU
	- Globale nel caso di dispositivi con GPU
Sia per l'esecuzione, sia per la trasmissione supponiamo un servizio deterministico.

Nel nostro caso, possiamo supporre:
- Ripartizione temporale della risorsa di calcolo
- Ripartizione PS della risorsa di trasmissione
Arrivando al seguente tipo di modello:
![[drawio/diagram_20261013112755.drawio.svg]]

### Ripartizione temporale della risorsa di calcolo
In questo caso, supponiamo che:
- Ad un certo istante di tempo, un unico (sotto-)modello possa eseguire
- Ci siano più code di priorità all'interno del server
- Supponiamo uno scheduling a $P$ classi con priorità non preemptive
	- Richieste che vanno nella stessa coda di priorità non eseguono per forza lo stesso (sotto-)modello, ma hanno semplicemente la stessa priorità di servizio associata.
- Il tempo di esecuzione di un (sotto-)modello sia modellabile come deterministico

#### Modello coda M/D/1 con priorità non preemptive
Supponiamo di avere $P$ classi; il tempo di coda osservato da una richiesta di classe $c$ in cui convergono richieste in un insieme $J$ è dato da:
$$
W_{c} = \frac{\sum_{j \in J} \lambda_j \; D_j^2}{(1 - \rho^{<c}) (1 - \rho^{\le c})}
$$
Con:
$$
\rho^{<c} = \sum_{h < c} \rho_h \qquad \rho^{\le c} = \sum_{h \le c} \rho_h
$$


### Ripartizione PS della risorsa di trasmissione
Possiamo supporre che le richieste siano gestite per l'invio secondo una disciplina processor-sharing.

#### Modello PS
Dato un job di tipo $j$ con tempo di servizio $S_j$, il tempo medio di risposta $T_j$ per il servizio in processor sharing è dato da:
$$
\mathbb{E}[T_j] = \frac{S_j}{1- \rho}
$$

Con:
$$
\rho = \sum_{j \in J} \lambda_j \ S_j
$$

# Modello matematico

## Modello di base
Sia dato un insieme di flussi $\mathcal{F}$; per ogni flusso definiamo:
- $\Lambda^f$ il tasso di generazione delle richieste per il flusso
	- $\{\hat{R}^f, \hat{E}^f, \hat{A}^f\}$ rispettivamente i requisiti sul tempo medio di risposta, energia media consumata e accuratezza media delle risposte
- $\mathcal{M}^f$ Insieme dei modelli tra cui si può scegliere per il flusso $f$
	- Questi modelli vengono decisi a priori in funzione del requisito di accuratezza. Infatti, visto che non ci sono decisioni che influenzano l'accuratezza del modello, possiamo pre-selezionare i modelli.

Dato un flusso $f \in \mathcal{F}$, consideriamo un modello $m \in \mathcal{M}^f$; questo modello può essere rappresentato come un DAG $G_m^f = (V_m^f, E_m^f, T_m^f)$, dove:
- $T_m^f$ rappresenta l'insieme dei tensori scambiati nel modello, insieme alle informazioni ad essi relative; in particolare:
- Indichiamo con $B^f_m$ la batch size del modello $m$

Indichiamo con:
$$
\mathcal{D} = \bigcup_{f \in \mathcal{F}} d^f \qquad \mathcal{E} \qquad \mathcal{C} \qquad \mathcal{S} = \mathcal{D} \cup \mathcal{E} \cup \mathcal{C}
$$
Rispettivamente:
- l'insieme dei dispositivi device, 
- l'insieme dei server edge.
- l'insieme dei server cloud.
- l'insieme complessivo dei dispositivi nel sistema

Indichiamo quindi con:
$$
\mathcal{S}^f = \{d^f\} \cup \mathcal{E} \cup \mathcal{C} 
$$
L'insieme dei dispositivi che il flusso $f$ può usare per la sua computazione.

Dato un server $s \in \mathcal{S}$, definiamo:
- $M^{main}_s$ la memoria principale disponibile sul server
- $M^{dev}_s$ la memoria del device di esecuzione se disponibile sul server

Consideriamo $K$ il massimo numero di componenti in cui vogliamo dividere il modello: queste componenti sono insiemi di nodi contigui (come definito in precedenza), quindi differenza di due ideals.

Dato il flusso $f \in \mathcal{F}$, siano definite le seguenti variabili binarie di base (altre seguiranno dopo dalle linearizzazioni e dalla gestione di altri aspetti):
- $x^f_{m, l, k} = 1$ se e solo se il livello $l \in V_m^f$ (con $m \in \mathcal{M}^f$) va sul blocco $k$
	- Binaria
- $e^{f}_{k,s} = 1$ se e solo se la componente $k$ esegue su server $s \in S^f$
	- Da rendere continua in $[0,1]$
- $u^f_{m} = 1$ se e solo se il flusso usa il modello $m \in \mathcal{M}^f$
	- Da rendere continua in $[0,1]$
- $c^f_{m,t,k} = 1$ se se solo se il tensore $t \in T_m^f$ del modello $m \in \mathcal{M}^f$ attraversa il taglio da componente $k$ a componente $k+1$
	- Da rendere continua in $[0,1]$
- $r^f_{k,s,p} = 1$ se e solo se alla componente $k$ del flusso $f$ viene assegnata priorità $p$ sul server $s$
	- Binaria

Partendo da queste, possiamo definire due relazioni lineari aggiuntive:
- $A_k^f$ , che indica se la componente $k$ viene mandata in esecuzione su qualche server (è attiva)
- $L_k^f$, che indica se la componente $k$ è l'ultima componente attiva per il flusso (come si vede successivamente, la successione delle componenti attive è prefissa)

Stando a queste definizioni, possiamo definire queste combinazioni nel seguente modo:
$$
\begin{align}
\forall f \in \mathcal{F} \qquad A_k^f \in [0,1] \qquad
& A_k^f =
\begin{cases}
\sum_{s \in \mathcal{S^f}}e_{k,s}^f \qquad & \forall k \in \mathcal{K}\\
0 \qquad & k = K+ 1
\end{cases}
\\ \\
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K} \qquad 
& L_k^f = A^f_k - A^f_{k+1}
\end{align}
$$

## Vincoli di coerenza 
### Vincoli di coerenza degli assegnamenti
- Per ogni flusso, viene scelto un unico modello
$$
\forall f \in \mathcal{F} \qquad \sum_{m \in \mathcal{M}^f} u_{m}^f = 1
$$

- Per ogni flusso e per ogni modello del flusso, i livelli del modello devono avere una componente se e solo se sono i livelli del modello usato
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall l \in V^f_m 
\qquad
\sum_{k \in \mathcal{K}} x^f_{m,l,k} = u^f_m
$$

- Per ogni flusso, una componente deve essere attiva se ha almeno un livello di un modello assegnato
$$
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K} \qquad
\begin{cases}
A^f_k \ge x^f_{m,l,k} \quad \forall m \in \mathcal{M}^f, \forall l \in V^f_m \\
A^f_k \le \sum_{m \in \mathcal{M}^f} \sum_{l \in V_m^f} x^f_{m,l,k}
\end{cases}
$$


- Per ogni flusso, per ogni modello, i livelli di input e di output, se il modello è usato, sono posizionati rispettivamente sulla prima e sull'ultima componente
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall v \in I^f_{m} \qquad x^f_{m,v,1} = u^f_m
$$
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall v \in O^f_m \quad \forall k \in \mathcal{K} \qquad
x^f_{m,v,k} \ge u_m^f + L_k^f - 1
$$

- Per ogni flusso, la prima e l'ultima componente sono posizionate sul device
$$
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K} \qquad
e^f_{1,d^f} = 1 
\quad \wedge \quad
e^f_{k,d^f} \ge L_k^f
$$


- Per ogni flusso, due componenti successive non possono andare sullo stesso server (conviene fonderle)
$$
\forall f \in \mathcal{F} \quad 
\forall k \in \mathcal{K} \ \backslash \ \{K\} \quad 
\forall s \in \mathcal{S}^f \qquad 
e^f_{k,s} + e^f_{k+1,s} \le 1
$$
	- **Rendere opzionale questo vincolo**: infatti, mantenere le componenti separate potrebbe permettere l'esecuzione in pezzi di modelli che altrimenti potrebbero non entrare all'interno del singolo server, oppure cambiare le priorità assegnate
	- Per ora, non aggiungerlo

### Vincoli di coerenza di flusso di esecuzione
- Per ogni flusso, la successione di componenti attive è prefissa
$$
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K}  \qquad A^f_{k+1} \le A^f_{k}
$$
	- Questo vincolo permette di dire che la successione di componenti attive non ha "buchi". Potrebbe essere utile per usare la soluzione del problema con $K$ componenti come cold start per la soluzione del problema con $K+1$ componenti

In questo modo, non ci possono essere situazioni come la *chain 1* in figura, ma tutte le successioni saranno come la *chain 2*
![[diagram_20260624134334.drawio.svg]]

- Per ogni flusso, per ogni modello, se un livello è assegnato ad una componente, i suoi predecessori devono essere stati assegnati entro quella componente.
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall (u,v) \in E_m^f \quad \forall k \in \mathcal{K} \qquad
\sum_{h = 1}^k x^f_{m,v,h} \le \sum_{h = 1}^{k} x_{m,u,h}
$$


Questo permette di evitare situazioni come quelle di seguito:
- Se la componente azzurra è $c$ e quella verde $c+1$, allora la condizione è violata, perché un successore di $g$ è in una componente con indice precedente
- Viceversa, la condizione è violata perché un successore di $a$ è in una componente precedente
![[diagram_20260624133518.drawio.svg]]


- Per ogni flusso, per ogni modello, un tensore del modello attraversa il taglio da componente $k$ a componente $k+1$ se e solo la sorgente del tensore è prima del taglio (quindi è assegnata ad una componente in $[1,k]$) e se c'è una destinazione del tensore dopo il taglio (quindi almeno una destinazione è assegnata ad una componente $[k+1,K]$ )
$$
\begin{align}
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K} \backslash \{K\} \quad \forall m \in \mathcal{M}^f \quad \forall t \in T_m^f \qquad \\ \\
\begin{cases}
c^f_{m,t,k} \le \sum_{h = 1}^{k} x^f_{m,\text{src}(t),h} \\
c^f_{m,t,k} \le \sum_{v \in \text{Dst}(t)} \sum_{h = k+1}^{K} x^f_{m,v,h} \\\
c^f_{m,t,k} \ge \sum_{h = 1}^{k} x^f_{m,\text{src}(t),h} + \sum_{h = k+1}^{K} x^f_{m,v,h} - 1 \qquad \forall v \in \text{Dst}(t)
\end{cases}
\end{align}
$$

## Tempo di risposta
### Tempo di batch fill
Dato un flusso $f \in \mathcal{F}$ e dato un modello $m \in \mathcal{M}^f$, le richieste vengono servite in batch $B^f_m$. Prima di far partire l'elaborazione, è necessario arrivi un numero di richieste pari alla dimensione del batch stesso.
![[diagram_20260622084856.drawio.svg]]
I server nella parte di inferenza non vedranno quindi un tasso di arrivo pari al tasso originale, ma un tasso di arrivo $\lambda^f_m$ dipendente dalla dimensione del batch del modello.

Considerando un modello $m$, il suo tempo di fill è:
- Caso medio.
  Tempo medio di attesa di una richiesta nel batch; richiesta $i$-esima aspetta $B^f_m-i$ richieste (per $i = 1, ..., B^f_m$)
$$
T^{f-fill}_m = \frac{1}{B^f_m} \sum_{k = 0}^{B^f_m - 1} \frac{k}{\Lambda^f} = \dots = \frac{B^f_m - 1}{2\Lambda^f}
$$
- Caso peggiore.
  Corrisponde al tempo di attesa della prima richiesta del batch, che dovrà aspettare le restanti $B^f_m - 1$ richieste
$$
T^{f-fill}_m = \frac{B^f_m - 1}{\Lambda^f}
$$

Da qui in avanti consideriamo il caso medio, quindi:
$$
T^{f-fill}_m  = \frac{B^f_m - 1}{2\Lambda^f}
$$

Inoltre, definiamo il tasso di arrivo riscalato alla dimensione del batch del modello come:
$$
\lambda^f_m = \frac{\Lambda^f}{B^f_m}
$$
Supponiamo comunque che il processo di arrivo si mantenga di Poisson. Possiamo vedere questo come una sorta di meccanismo di backpressure, per cui il sistema regola, scegliendo il modello, il tasso di invio dell'utente.

In realtà, questo tempo dipende dal modello selezionato, ed in particolare dalla dimensione del suo batch; da cui abbiamo che il tempo di fill per il flusso $f$ è:
$$
T^{f-fill} = \sum_{m \in \mathcal{M^f}} u^f_m \ \frac{B^f_m - 1}{2\Lambda^f}
$$

### Tempo di esecuzione di una componente
Il tempo di esecuzione di una componente $k$ su un server $s$ è dato da:
$$
T^{f-ex}_{m,k,s} = \sum_{l \in V_m^f} \tau^f_{m,l,s} \cdot x^f_{m,l,k} \cdot e^f_{k,s}
$$

Il tempo di esecuzione di una componente sarà quindi dato da:
$$
T^{f-ex}_{k} = \sum_{m \in \mathcal{M}^f} \sum_{s \in \mathcal{S}^f} T^{f-ex}_{m,k,s}
$$

#### Linearizzazione
Definiamo $z^f_{m,l,k,s} = x^f_{m,l,k} \cdot e^f_{k,s} \in [0,1]$ ; questa è vincolata a:
$$
\begin{cases}
z^f_{m,l,k,s} \le x^f_{m,l,k} \\
z^f_{m,l,k,s} \le e^f_{k,s} \\
z^f_{m,l,k,s} \ge x^f_{m,l,k} + e^f_{k,s} - 1
\end{cases}
$$

### Tempo di trasmissione
Definita la quantità totale di dati che passa il taglio come:
$$
\Delta^f_{m,k} = \sum_{t \in T^f_m} \delta_t \cdot c^f_{m,t,k}
$$
Notare che se il modello non è scelto, questo valore sarà nullo, in quanto i livelli del modello non saranno assegnati a nessuna componente.

Possiamo scrivere il tempo necessario alla trasmissione di un unico pacchetto contenente tutti i tensori del contesto, come:
$$
T^{f-tx}_{m,k,s} = \sum_{s^{'} \in \mathcal{S}^f} \left(
\frac{\Delta^f_{m,k}}{\beta_{s s^{'}}} 
\cdot e^f_{k,s} \cdot e^f_{k+1,s^{'}}
\right)
$$
Mentre la latenza di trasmissione per la componente $k$ dal server $s$ è data da:
$$
\xi_{k,s}^f = \sum_{s^{'} \in \mathcal{S}^f}
\left(
\xi_{s s^{'}} \cdot e^f_{k,s} \cdot e^f_{k+1,s^{'}}
\right)
$$

Notare come sia necessario separare la latenza tra i due server dalla trasmissione del contesto: il motivo è che la latenza è indipendente dal modello, quindi dobbiamo considerarla una sola volta.

#### Linearizzazione
Definiamo la variabile ausiliaria $q^f_{k,s,s^{'}} = e^f_{k,s} \cdot e^f_{k+1,s^{'}} \in \{0, 1\}$:
$$
\begin{aligned}
q^f_{k,s,s^{'}} = e^f_{k,s} \cdot e^f_{k+1,s^{'}} 
& \quad \text{ s.t. } \quad
\begin{cases}
q^f_{k,s,s^{'}} \le e^f_{k,s} \\
q^f_{k,s,s^{'}} \le e^f_{k+1,s^{'}} \\
q^f_{k,s,s^{'}} \ge e^f_{k,s} + e^f_{k+1,s^{'}} -1 \\
\end{cases}
\end{aligned}
$$

##### Con massimo
$$
\begin{aligned}
& \hat{\Delta}^f_{m,k,s,\beta,s^{'}} = \Delta^{f}_{m,k} \cdot q^f_{k,s,s^{'}}
& \quad \text{ s.t. } \quad
\begin{cases}
\hat{\Delta}^f_{m,k,s,\beta,s^{'}} \ge 0 \\
\hat{\Delta}^f_{m,k,s,\beta,s^{'}} \le \Delta^f_{m,k} \\
\hat{\Delta}^f_{m,k,s,\beta,s^{'}} \le \Delta^{f-max}_{m} \cdot q^f_{k,s,s^{'}} \\
\hat{\Delta}^f_{m,k,s,\beta,s^{'}} \ge \Delta^f_{m,k} - \Delta^{f-max}_{m} \cdot (1-q^f_{k,s,s^{'}}) \\
\end{cases}
\\ \\
& \Delta^{f-max}_{m} = \sum_{t \in T^f_m} \delta_t

\end{aligned}
$$


> [!NOTE] Pre-calcolo del $\Delta^{f-max}_{m}$
> In realtà, questo valore può essere pre-calcolato in modo abbastanza esatto. La massima quantità di dati che è possibile trasmettere attraverso un taglio non dipende dal $k$.
> Di conseguenza, generando tutti i possibili ideals del DAG del modello, si può calcolare quanto viene trasmesso attraverso un taglio.

##### Con indicator constraint
Definiamo la variabile ausiliaria:
$$
T^{f,\mathrm{tx}}_{m,k,s} \ge 0
$$

Imponiamo i seguenti indicator constraints:
$$
\forall k \in \mathcal{K}
\qquad
\forall s,s' \in \mathcal{S}^f
\qquad \qquad
q^f_{k,s,s'} = 1
\quad\Longrightarrow\quad
\beta_{ss'}\,T^{f,\mathrm{tx}}_{m,k,s}=\Delta^f_{m,k}
$$

Per azzerare il tempo quando il server sorgente non ospita la componente o quando non esiste una componente successiva, imponiamo:
$$
0 \le T^{f-tx}_{m,k,s} \le \hat{R}^f \cdot (1 - \sum_{s^{'} \in \mathcal{S}^f}q^f_{k,s,s^{'}})
$$

Dato che ogni componente attiva è assegnata a un unico server, per ogni taglio attivo esiste una sola coppia $(s,s')$ con $q^f_{k,s,s'}=1$. In quella coppia:
$$
T^{f,\mathrm{tx}}_{m,k,s}=\frac{\Delta^f_{m,k}}{\beta_{ss'}}.
$$

Per i modelli non selezionati, il tempo è nullo, perché $\Delta^f_{m,k}=0$.



### Coda di esecuzione - Prio non preemptive
Indichiamo con $P$ il massimo numero di code di priorità che un server $s$ può gestire (supposto uguale per tutti i server, ma non è un vincolo necessario); supponiamo che le code siano a priorità decrescente ($1 \rightarrow 2 \rightarrow 3 \rightarrow \dots$).

Una classe di richieste assegnata al server è identificata dalla coppia $(f,k)$: una richiesta che arriva al server è data dal flusso e dalla componente relativa, a prescindere dal modello.

Indichiamo con:
- $n_{s,p} = 1$ se e solo se la coda a priorità $p$ è usata su $s$
  - Da rendere continua in $[0,1]$

Indichiamo con $\mathcal{P}=\{1,\ldots,P\}$ l'insieme degli indici di priorità.

I vincoli di coerenza principali qui sono:
- Se una classe è mandata su un server le deve essere data esattamente una coda:
$$
  \forall f \in \mathcal{F},\ \forall k \in \mathcal{K},\
  \forall s \in \mathcal{S}^f
  \qquad
  \sum_{p \in \mathcal{P}} r^f_{k,s,p} = e^f_{k,s}
  $$

- Una coda a priorità $p$ è usata se almeno una classe è mandata su di essa:
$$
  \forall s \in \mathcal{S},\ \forall p \in \mathcal{P}
  \qquad
  \begin{cases}
  n_{s,p} \le
  \displaystyle\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
  \sum_{k \in \mathcal{K}} r^f_{k,s,p}
  \\[8pt]
  n_{s,p} \ge r^f_{k,s,p}
  \qquad
  \forall f \in \mathcal{F}: s \in \mathcal{S}^f,\
  \forall k \in \mathcal{K}
  \end{cases}
  $$

- L'uso delle code di priorità è prefisso (non indispensabile, ma per non lasciare code intermedie vuote)
  $$
  \forall s \in \mathcal{S},\ \forall p = 1,\ldots,P-1
  \qquad
  n_{s,p} \ge n_{s,p+1}
  $$

La formula di code con priorità in generale è la seguente:
$$
W_{s,p}
=
\frac{H_s}
{2(1-\rho_s^{<p})(1-\rho_s^{\le p})}
$$
con:
$$
H_s =
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{m \in \mathcal{M}^f}
\sum_{k \in \mathcal{K}}
\lambda_m^f
\left(T_{m,k,s}^{f,\mathrm{ex}}\right)^2
$$

Dobbiamo esprimere le variabili $\rho_s^{<p}$ perché il loro valore varia in funzione dell'assegnazione delle classi alle code. Definiamo prima di tutto:

$$
\rho^f_{k,s}
=
\sum_{m \in \mathcal{M}^f}
\lambda^f_m T_{m,k,s}^{f,\mathrm{ex}}
$$

Da cui troviamo:
$$
\rho_s^{<p}
=
\sum_{h=1}^{p-1}
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
r^f_{k,s,h}\,\rho^f_{k,s}
$$
$$
\rho_s^{\le p}
=
\sum_{h=1}^{p}
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
r^f_{k,s,h}\,\rho^f_{k,s}
$$

#### Linearizzazione delle utilizzazioni
Dobbiamo esprimere il prodotto:

$$
\bar{\rho}^f_{k,s,p}
=
r^f_{k,s,p}\,\rho^f_{k,s}
$$

Poiché ogni classe può essere assegnata a una sola coda di priorità sul server (una sola delle variabili $r^f_{k,s,p}$ è attiva), possiamo sfruttare il vincolo di assegnamento già definito:
$$
\sum_{p \in \mathcal{P}}r^f_{k,s,p}=e^f_{k,s}
$$

Introduciamo quindi le variabili continue $\bar{\rho}^f_{k,s,p}$ con i seguenti vincoli:
$$
\forall f \in \mathcal{F},\
\forall k \in \mathcal{K},\
\forall s \in \mathcal{S}^f
\qquad
\begin{cases}
0 \le \bar{\rho}^f_{k,s,p}
\le \rho^{\max}r^f_{k,s,p}
\qquad \forall p \in \mathcal{P}
\\[8pt]
\displaystyle\sum_{p \in \mathcal{P}}
\bar{\rho}^f_{k,s,p}
=
\rho^f_{k,s}
\end{cases}
$$

Il valore $\rho^{\max}$ costituisce un upper bound valido per l'utilizzazione di una singola classe, poiché imponiamo:

$$
\rho_s
=
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
\rho^f_{k,s}
\le \rho^{\max}<1
$$

Inoltre che $\rho^f_{k,s}=0$ quando la componente non è assegnata al server, per come è definito il tempo di esecuzione della componente.

La formulazione è esatta: poiché soltanto una priorità può essere selezionata, tutte le altre variabili $\bar{\rho}^f_{k,s,p}$ saranno nulle e quella corrispondente alla priorità attiva assumerà il valore $\rho^f_{k,s}$.

#### Discretizzazione di $\rho_s^{<p}$
Supponiamo di dividere l'intervallo $[0,\rho^{\max}]$ in $I$ intervalli. Facciamo in modo che $\rho_s^{<p}$ assuma un valore fisso all'interno di questo intervallo; poiché vogliamo maggiorare, ci interessa l'estremo superiore dell'intervallo.

Definiamo quindi:

$$
0=\hat{\rho}_0
<
\hat{\rho}_1
<
\hat{\rho}_2
<
\dots
<
\hat{\rho}_I=\rho^{\max}
$$

Definiamo la variabile $g_{s,p,i}=1$ se e solo se il valore $\rho_s^{<p}$ ricade nell'intervallo $i$-esimo.

Poiché per la prima priorità si ha:
$$
\rho_s^{<1}=0
$$

non è necessario discretizzarne l'utilizzazione cumulativa. La discretizzazione viene quindi applicata soltanto alle priorità $p=2,\ldots,P$.

Di conseguenza, le variabili $g_{s,p,i}$ e i relativi vincoli di discretizzazione sono definiti soltanto per:
$$
p=2,\ldots,P,\qquad i=1,\ldots,I
$$

I vincoli di coerenza qui sono i seguenti:
- Data una coda, essa ha esattamente un bin attivo.
  Imponiamo in questo modo, a prescindere che la coda sia usata o meno, perché anche se una coda non è usata, supposto di porsi in quella coda, comunque il valore di $\rho_s^{<p}$ visto da quella coda è non nullo.
  $$
  \forall s \in \mathcal{S},\ \forall p=2,\ldots,P
  \qquad
  \sum_{i=1}^{I}g_{s,p,i}=1
  $$

- Il valore reale deve cadere nel bin associato
$$
  \forall s \in \mathcal{S},\ \forall p=2,\ldots,P
  \qquad
  \sum_{i=1}^{I}\hat{\rho}_{i-1}g_{s,p,i}
  \le
  \rho_s^{<p}
  \le
  \sum_{i=1}^{I}\hat{\rho}_i g_{s,p,i}
  $$

###### Errore relativo massimo
L'errore è dato da:
$$
\frac{\hat{W}_{s,p}}{W_{s,p}}
=
\frac{1-\rho_s^{<p}}{1-\hat{\rho}_i}
$$

Alla peggio, il $\rho_s^{<p}=\hat{\rho}_{i-1}$
$$
\frac{\hat{W}_{s,p}}{W_{s,p}}
=
\frac{1-\hat{\rho}_{i-1}}{1-\hat{\rho}_i}
$$

L'errore relativo massimo è dato da:
$$
\epsilon_i^{\max}
=
\frac{1-\hat{\rho}_{i-1}}{1-\hat{\rho}_i}-1
$$

Da cui, se per tutti i bucket voglio errore relativo sotto un certo $\epsilon$:
$$
\hat{\rho}_i
\le
1-\frac{1-\hat{\rho}_{i-1}}{1+\epsilon}
$$

Prendendo i bucket più larghi possibili abbiamo:
$$
\begin{aligned}
1-\hat{\rho}_i
&=
\frac{1}{1+\epsilon}(1-\hat{\rho}_{i-1})
\\[8pt]
\eta_i
&=
\frac{1}{1+\epsilon}\eta_{i-1},
\qquad
\eta_i=1-\hat{\rho}_i
\\[12pt]
&\text{Questa è una serie geometria; per }i=I\text{ abbiamo}
\\[12pt]
1-\hat{\rho}_I
&=
\frac{1}{(1+\epsilon)^I}
\end{aligned}
$$

Risolvendo in $I$ e cambiando base al logaritmo troviamo che il minimo numero di bucket per avere errore più piccolo di $\epsilon$ è:
$$
I=
\left\lceil
-\frac{\log(1-\rho^{\max})}{\log(1+\epsilon)}
\right\rceil
$$

Gli estremi degli intervalli vengono quindi definiti come:
$$
\hat{\rho}_i=
\begin{cases}
1-\displaystyle\frac{1}{(1+\epsilon)^i}
& i=0,\ldots,I-1
\\[10pt]
\rho^{\max}
& i=I
\end{cases}
$$

Con l'ultimo estremo troncato a $\rho^{\max}$.

#### Rappresentazione conica
Definiamo $h_{s,p,i}$, tale che:

$$
\forall s \in \mathcal{S},\
\forall p=2,\ldots,P,\
\forall i=1,\ldots,I
\qquad
\begin{cases}
h_{s,p,i}\ge0
\\[6pt]
h_{s,p,i}\le M_s g_{s,p,i}
\end{cases}
$$

Con il seguente valore che maggiora il tempo di coda visto da qualsiasi coda sul server $s$:

$$
M_s
=
\frac{
\displaystyle\sum_{\substack{f\in\mathcal{F}:\\s\in\mathcal{S}^f}}
\max_{m\in\mathcal{M}^f}
\left\{
\lambda_m^f
\left(\bar{T}_{m,s}^f\right)^2
\right\}
}{
2(1-\rho^{\max})^2
}
$$

Con $\bar{T}_{m,s}^f$ la somma dei tempi di esecuzione dei livelli del modello $m$ di $f$ su $s$; infatti, alla peggio, il modello è in un'unica componente che viene messa su $s$. Più in generale:

$$
\bar{T}_{m,s}^f
=
\sum_{l\in V_m^f}\tau_{m,l,s}^f
$$

$$
\sum_{k\in\mathcal{K}}
\left(T_{m,k,s}^{f,\mathrm{ex}}\right)^2
\le
\left(
\sum_{k\in\mathcal{K}}T_{m,k,s}^{f,\mathrm{ex}}
\right)^2
\le
\left(\bar{T}_{m,s}^f\right)^2
$$

Per la prima priorità, non essendo necessaria la discretizzazione, possiamo rappresentare direttamente il tempo medio di attesa mediante la variabile continua $h_{s,1}\ge0$, imponendo:

$$
\forall s\in\mathcal{S}
\qquad
2h_{s,1}(1-\rho_s^{\le1})\ge H_s
$$

Con il seguente vincolo superiore:

$$
\forall s\in\mathcal{S}
\qquad
0\le h_{s,1}\le M_s
$$

Per le priorità successive $p=2,\ldots,P$, manteniamo invece la rappresentazione discretizzata. Definiamo:

$$
\hat{h}_{s,p}
=
\sum_{i=1}^{I}
(1-\hat{\rho}_i)h_{s,p,i}
$$

Di queste variabili, soltanto una sarà attiva, cioè quella corrispondente al bucket selezionato.

Imponiamo quindi:

$$
\forall s\in\mathcal{S},\ \forall p=2,\ldots,P
\qquad
2\hat{h}_{s,p}(1-\rho_s^{\le p})\ge H_s
$$

Questa rappresentazione costituisce un vincolo conico ruotato, dato che entrambi i fattori a sinistra sono non negativi e il membro destro è una somma di quadrati pesati.

Nella definizione di $\hat{h}_{s,p}$, sopravvive soltanto il termine del bucket attivo. Infatti, se $j$ è l'indice del bucket attivo, troviamo:

$$
\begin{aligned}
&2(1-\hat{\rho}_j)\,h_{s,p,j}
(1-\rho_s^{\le p})
\ge H_s
\\[12pt]
&h_{s,p,j}
\ge
\frac{H_s}{
2(1-\hat{\rho}_j)(1-\rho_s^{\le p})
}
\end{aligned}
$$

Quindi $h_{s,p,j}$ maggiora il tempo medio di attesa della coda $p$, utilizzando l'estremo superiore del bucket selezionato per approssimare $\rho_s^{<p}$.

Definendo:

$$
h_{s,p}
=
\sum_{i=1}^{I}h_{s,p,i}
$$

otteniamo il tempo di attesa maggiorato associato alla priorità $p$ sul server $s$.


#### Tempo di coda di esecuzione di una componente
Definiamo quindi il tempo medio di attesa in coda della classe $(f,k)$:
$$
w^{f,\mathrm{ex-enq}}_k
$$

Possiamo imporre:

$$
w^{f,\mathrm{ex-enq}}_k
\ge
h^s_p - M_s (1-r^f_{k,s,p})
\qquad
\forall s \in \mathcal{S}^f,\ \forall p \in \mathcal{P}
$$

Se la classe $(f,k)$ è assegnata alla priorità $p$ sul server $s$, il vincolo diventa:
$$
w^{f,\mathrm{ex-enq}}_k \ge h^s_p
$$
Viceversa, il vincolo viene disattivato tramite $M_s$.

Per garantire che il tempo di attesa sia nullo quando la componente non è attiva, definiamo:
$$
\bar{M}_f = \max_{s \in \mathcal{S}^f} M_s
$$

e imponiamo:

$$
0 \le w^{f,\mathrm{ex-enq}}_k
\le \bar{M}_f \, A^f_k
$$

In questo modo, il tempo di attesa della componente è rappresentato mediante un'unica variabile continua, indipendente dal server e dal modello selezionato.

La formulazione garantisce che tale variabile maggiori il tempo di attesa della priorità effettivamente assegnata. L'uguaglianza si ottiene quando il tempo di attesa viene minimizzato o quando i vincoli successivi rendono stretta la maggiorazione.


### Coda di trasmissione - Processor Sharing
Supponiamo di modellare la trasmissione su un server come processor sharing.

Definiamo il $\rho_{s}^{tx}$ come:
$$
\rho_s^{tx} =
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{m \in \mathcal{M}^f}
\sum_{k \in \mathcal{K}}
\lambda^f_m \cdot T^{f-tx}_{m,k,s}
$$

Il tempo medio di risposta per la trasmissione del risultato della componente $k$ in Processor Sharing è dato da:
$$
R^{f-tx}_{k} = \sum_{m \in \mathcal{M}^f} \sum_{s \in \mathcal{S}^f}\frac{T^{f-tx}_{m,k,s}}{1 - \rho^{tx}_s}
$$
Dove il denominatore può essere gestito con lo stesso principio del denominatore nel caso precedente.

Definiamo inoltre la latenza per la trasmissione del risultato di una componente come:
$$
\xi^f_{k} = \sum_{s \in \mathcal{S}^f} \xi_{k,s}^f
$$

Imponiamo inoltre il vincolo di stabilità:
$$
\rho_s^{tx} \le \rho^{tx-max} < 1
\qquad \forall s \in \mathcal{S}
$$


Poiché il denominatore di $R^{f-tx}_k$ dipende dalle variabili decisionali, questa espressione non è direttamente rappresentabile come vincolo MISOCP.

Introduciamo quindi una discretizzazione dell'utilizzazione della rete.

#### Discretizzazione di $\rho_s^{tx}$
Supponiamo di dividere l'intervallo $[0,\rho^{tx-max}]$ in $I$ intervalli.

Definiamo:

$$
0=\hat{\rho}_0 <
\hat{\rho}_1 <
\hat{\rho}_2 <
\dots <
\hat{\rho}_I=\rho^{tx-max}
$$

Poiché vogliamo maggiorare il tempo medio di risposta, utilizziamo l'estremo superiore dell'intervallo selezionato.

Definiamo la variabile binaria:
$$
g^{tx}_{s,i} =
\begin{cases}
1 & \text{se il bin } i \text{ è selezionato per il server }s\\
0 & \text{altrimenti}
\end{cases}
$$

I vincoli di coerenza sono i seguenti:
- Ogni server ha esattamente un bin attivo:
$$
\forall s \in \mathcal{S}
\qquad
\sum_{i=1}^{I}g^{tx}_{s,i}=1
$$

- L'utilizzazione reale deve appartenere all'intervallo selezionato:
$$
\forall s \in \mathcal{S}
\qquad
\sum_{i=1}^{I}\hat{\rho}_{i-1}g^{tx}_{s,i}
\le
\rho_s^{tx}
\le
\sum_{i=1}^{I}\hat{\rho}_i g^{tx}_{s,i}
$$

Definiamo quindi l'utilizzazione discretizzata come:
$$
\widetilde{\rho}_s^{tx}
=
\sum_{i=1}^{I}\hat{\rho}_i g^{tx}_{s,i}
$$

Per costruzione:
$$
\rho_s^{tx} \le \widetilde{\rho}_s^{tx}
\le \rho^{tx-max}<1
$$

##### Errore relativo massimo
Consideriamo il bin $i$, identificato dall'intervallo:
$$
[\hat{\rho}_{i-1},\hat{\rho}_i]
$$

Il rapporto tra il tempo di risposta discretizzato e quello esatto è:
$$
\frac{\widetilde{R}}{R}
=
\frac{1-\rho_s^{tx}}{1-\hat{\rho}_i}
$$

Il rapporto massimo si verifica quando l'utilizzazione reale coincide con l'estremo inferiore del bin:
$$
\rho_s^{tx}=\hat{\rho}_{i-1}
$$

Otteniamo:
$$
\epsilon_i^{max}
=
\frac{1-\hat{\rho}_{i-1}}{1-\hat{\rho}_i}-1
$$

Volendo garantire un errore relativo massimo $\epsilon$ per ogni intervallo, imponiamo:
$$
\frac{1-\hat{\rho}_{i-1}}{1-\hat{\rho}_i}
\le 1+\epsilon
$$

Da cui:
$$
\hat{\rho}_i
\le
1-\frac{1-\hat{\rho}_{i-1}}{1+\epsilon}
$$

Prendendo gli intervalli più ampi consentiti, otteniamo:
$$
1-\hat{\rho}_i
=
\frac{1}{1+\epsilon}
(1-\hat{\rho}_{i-1})
$$

Si tratta di una progressione geometrica:
$$
1-\hat{\rho}_i
=
\frac{1}{(1+\epsilon)^i}
$$

Il numero minimo di intervalli necessario per coprire $[0,\rho^{tx-max}]$ rispettando il limite di errore è:
$$
I=
\left\lceil
-\frac{\log(1-\rho^{tx-max})}
{\log(1+\epsilon)}
\right\rceil
$$

Gli estremi degli intervalli vengono quindi definiti come:
$$
\hat{\rho}_i =
\begin{cases}
1-\displaystyle\frac{1}{(1+\epsilon)^i}
& i=0,\dots,I-1\\[8pt]
\rho^{tx-max}
& i=I
\end{cases}
$$

L'ultimo intervallo viene eventualmente troncato per terminare esattamente in $\rho^{tx-max}$.



#### Con indicator constrints
Definiamo quindi la variabile continua non negativa:
$$
R^{f-tx}_k \ge 0
$$

che rappresenta una maggiorazione del tempo medio di risposta della trasmissione relativa alla componente $k$ del flusso $f$.

Per ogni server $s$ e per ogni bin $i$, imponiamo il seguente indicator constraint:
$$
\begin{aligned}
&\forall f \in \mathcal{F},
\quad \forall k \in \mathcal{K},
\quad \forall s \in \mathcal{S}^f,
\quad \forall i=1,\dots,I
\\[6pt]
&g^{tx}_{s,i}=1
\implies
(1-\hat{\rho}_i)R^{f-tx}_k
\ge
\sum_{m \in \mathcal{M}^f}T^{f-tx}_{m,k,s}
\end{aligned}
$$

Il vincolo viene attivato soltanto per il bin selezionato sul server $s$.
Poiché ogni componente è assegnata a un solo server e viene selezionato un unico modello, soltanto il server sorgente effettivo presenta un tempo nominale di trasmissione non nullo.

Per gli altri server, il termine a destra dell'indicator constraint è nullo e il vincolo risulta automaticamente soddisfatto.

La formulazione utilizza esclusivamente indicator constraints su disuguaglianze lineari e rimane quindi compatibile con una formulazione MISOCP.

La variabile $R^{f-tx}_k$ costituisce un upper bound del valore ottenuto dalla formula PS. Il vincolo diventa stretto quando il modello minimizza tale variabile, direttamente o attraverso il tempo di risposta end-to-end.

## Modello della memoria
### Memoria di esecuzione
La memoria occupata dai pesi e dalle attivazioni sarà data da
$$
Q^{f-ex-wg}_{m,k,s} = \sum_{l \in V_m^f} \omega^f_{m,l} \cdot x^f_{m,l,k} \cdot e^f_{k,s}
\qquad \qquad
Q^{f-ex-act}_{m,k,s} = \sum_{l \in V_m^f} \alpha^f_{m,l} \cdot x^f_{m,l,k} \cdot e^f_{k,s}
$$
Questa relazione suppone, nel caso delle attivazioni, che tutti i pesi dei modelli mandati in deployment siano in memoria e che tutte le attivazioni sono in memoria durante l'esecuzione di una componente.

Per il modello che abbiamo considerato, possiamo supporre che un solo modello sia attivo ad un certo istante, ma che tutti i pesi siano caricati. Da cui:
$$
Q^{ex-wg}_{s} = \sum_{f \in \mathcal{F}} \sum_{m \in \mathcal{M}^f} \sum_{k \in \mathcal{K}} Q^{f-ex-wg}_{m,k,s}
\qquad \qquad
Q^{ex-act}_{s} = \max_{f,m,k} Q^{f-ex-act}_{m,k,s}
$$

### Memoria occupata
Supponendo che la memoria dedicata sia nulla quando il server non ha una GPU, possiamo scrivere quanto segue:
$$
\begin{align}
Q^{ex-wg}_{s} + Q^{ex-act}_{s} & \le M_s^{GPU} \cdot 1_{\{M_s^{GPU} \neq 0\}} + M_s^{RAM} \cdot (1 -  1_{\{M_s^{GPU} \neq 0\}})
\end{align}
$$
Questo significa che se il server ha una GPU, l'unica cosa che pesa sulla memoria principale sono le code, mentre l'esecuzione del modello vero e proprio è su device. In caso contrario, il primo vincolo è banalmente vero e tutto quanto è spostato sulla memoria principale del server.

  
## Requisiti di Qualità
### Tempo medio di risposta
Il tempo medio di risposta per un flusso è dato da:
$$
R^f = T^{f-fill}  + \sum_{k = 1}^{K} \left(w_k^{f-ex-enq} + T_k^{f-ex} \right) + \sum_{k = 1}^{K-1} \left(R^{f-tx}_k + \xi^f_k \right)
$$
Con requisito:
$$
\forall f \in \mathcal{F} \qquad R^f \le \hat{R}^f
$$

## Obiettivo
Indichiamo con:
$$\Lambda = \sum_{f \in \mathcal{F}} \Lambda^f$$
Come funzioni obiettivo possiamo considerare le seguenti:
- Somma pesata sul tasso di arrivo dei flussi
$$
\begin{aligned}
& \sum_{f \in \mathcal{F}} \frac{\Lambda^f}{\Lambda} \cdot (R^f - \hat{R}^f)
\\ \\
\end{aligned}
$$
- Summa pesata dei rapporti
$$
\sum_{f \in \mathcal{F}} \frac{\Lambda^f}{\Lambda} \cdot \left( \frac{R^f}{\hat{R}^f} - 1 \right)
$$
