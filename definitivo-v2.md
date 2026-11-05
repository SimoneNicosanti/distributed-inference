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

![[diagram_20260615095322.drawio.svg]]
## Modello della rete
La rete di server può essere rappresentata mediante un grafo diretto $G_N = (V_N, E_N)$, dove:
- $V_N$ rappresenta l'insieme dei server
- $E_N$ rappresenta l'insieme dei collegamenti tra server; ogni arco ha associate informazioni relative a:
	- Banda disponibile sull'arco
	- Latenza di trasmissione

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
	- Nodi di input e nodi di output

### Divisione della DNN
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
W_{c} = \frac{\sum_{j \in J} \lambda_j \; D_j^2}{2(1 - \rho^{<c}) (1 - \rho^{\le c})}
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

## Modello della richiesta
Una richiesta di ottimizzazione per un certo flusso specifica quanto segue:
- Tasso di arrivo
- Requisiti 
- Eventualmente il dispositivo $d^f$ che si vuole usare
	- Rappresenta il dispositivo device per cui si può eseguire parte della computazione del flusso
	- La richiesta si può configurare in modo che il device sia relay-only; in questo caso gli unici livelli che possono essere messi sul device sono i livelli di input e di output del modello.
- Il worker di ingress $s^f_I$ e di sink $s^f_O$ che si vogliono usare per inviare input o per ricevere il risultato:
	- Se non specificati, la scelta di dove posizionarli è fatta dall'ottimizzatore

Questo permette di avere flessibilità verso diverse configurazioni:
- $d^f$ è impostato a relay-only; $s_I^f = d^f$ e $s_O^f = d^f$
	- Il device fa solo da punto di ingress e da punto di raccolta del risultato e nessun calcolo viene fatto sul device 
	- L'input viene immediatamente trasferito verso un altro server senza elaborazione
	- Semplifica sperimentazione cloud-only / server-only
- $d^f$ è impostato a compute; $s_I^f = d^f$ e $s_O^f = d^f$
	- Il device fa solo da punto di ingress e da punto di raccolta del risultato
	- Possibile fare calcoli sul device
- $s_I^f$ e $s^f_O$ non impostati
	- In questo caso sono scelti dall'ottimizzatore
	- Ci permette di modellare il caso in cui non siamo interessati a ricevere l'output (e.g., trasferimento verso uno storage)

# Modello matematico

## Modello di base
Sia dato un insieme di flussi $\mathcal{F}$; per ogni flusso definiamo:
- $\Lambda^f$ il tasso di generazione delle richieste per il flusso
	- $\{\hat{R}^f, \hat{E}^f, \hat{A}^f\}$ rispettivamente i requisiti sul tempo medio di risposta, energia media consumata e accuratezza media delle risposte
	- Device $d^f$ e se è in relay-only
	- Facoltativi: worker di ingress $s_I^f$ e worker di out $s^f_O$ 
- $\mathcal{M}^f$ Insieme dei modelli tra cui si può scegliere per il flusso $f$
	- Questi modelli vengono decisi a priori in funzione del requisito di accuratezza. Infatti, visto che non ci sono decisioni che influenzano l'accuratezza del modello, possiamo pre-selezionare i modelli.

Dato un flusso $f \in \mathcal{F}$, consideriamo un modello $m \in \mathcal{M}^f$; questo modello può essere rappresentato come un DAG $G_m^f = (V_m^f, E_m^f, T_m^f)$, dove:
- $T_m^f$ rappresenta l'insieme dei tensori scambiati nel modello, insieme alle informazioni ad essi relative:
	- Chiamiamo $\delta_t$ la dimensione del tensore
	- Indichiamo con $\text{src}(t)$ e $\text{Dst}(t)$ rispettivamente il nodo sorgente e l'insieme nodi destinazione 
	- Indichiamo con $\text{Dst}^{*}(t)$ l'insieme delle destinazioni massimali del tensore: una destinazione è massimale se nessun'altra destinazione di quel tensore è raggiungibile da essa.
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
Da qui, deriviamo il grafo di rete $G_N^f = (\mathcal{S}^f, E_N^f)$ come il grafo di rete del sistema ristretto ai nodi $\mathcal{S}^f$.


Dato un server $s \in \mathcal{S}$, definiamo:
- $M^{main}_s$ la memoria principale disponibile sul server
- $M^{dev}_s$ la memoria del device di esecuzione se disponibile sul server

Consideriamo $K$ il massimo numero di componenti in cui vogliamo dividere il modello: queste componenti sono insiemi di nodi contigui (come definito in precedenza).

Dato il flusso $f \in \mathcal{F}$, siano definite le seguenti variabili binarie di base (altre seguiranno dopo dalle linearizzazioni e dalla gestione di altri aspetti):
- $x^f_{m, l, k} = 1$ se e solo se il livello $l \in V_m^f$ (con $m \in \mathcal{M}^f$) va sul blocco $k$
	- Binaria
- $e^{f}_{k,s} = 1$ se e solo se la componente $k$ esegue su server $s \in S^f$
	- Per come vincolata successivamente, continua in $[0,1]$
- $u^f_{m} = 1$ se e solo se il flusso usa il modello $m \in \mathcal{M}^f$
	- Per come vincolata successivamente, continua in $[0,1]$
- $c^f_{m,t,k} = 1$ se se solo se il tensore $t \in T_m^f$ del modello $m \in \mathcal{M}^f$ attraversa il taglio da componente $k$ a componente $k+1$
	- Per come vincolata successivamente, continua in $[0,1]$
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
- Per ogni flusso, viene scelto un unico modello:
$$
\forall f \in \mathcal{F} \qquad \sum_{m \in \mathcal{M}^f} u_{m}^f = 1
$$

- Per ogni flusso e per ogni modello del flusso, i livelli del modello devono avere una componente se e solo se sono i livelli del modello usato:
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall l \in V^f_m 
\qquad
\sum_{k \in \mathcal{K}} x^f_{m,l,k} = u^f_m
$$

- Per ogni flusso e per ogni modello del flusso, una componente deve essere attiva se ha almeno un livello del modello assegnato:
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall k \in \mathcal{K} \qquad
\begin{cases}
A^f_k \ge x^f_{m,l,k} \quad  \forall l \in V^f_m \\
A^f_k + u^f_m - 1 \le \sum_{l \in V_m^f} x^f_{m,l,k}
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
	- Vincolo di rafforzamento per LP:
		- $\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall v \in O^f_m \quad \forall k \in \mathcal{K} \qquad x^f_{m,v,k} \le L_k^f$

- Per ogni flusso, la prima e l'ultima componente sono posizionate rispettivamente sul server di ingress e sul server di out (se specificati):
$$
\begin{align}
\forall f \in \mathcal{F} &\qquad e^f_{1,s_I^f} = 1 \\
\forall f \in \mathcal{F} \quad\forall k \in \mathcal{K} &\qquad e^f_{k,s_O^f} \ge L_k^f
\end{align}
$$

- Per ogni flusso, due componenti successive non possono andare sullo stesso server:
$$
\forall f \in \mathcal{F} \quad 
\forall k \in \mathcal{K} \ \backslash \ \{K\} \quad 
\forall s \in \mathcal{S}^f \qquad 
e^f_{k,s} + e^f_{k+1,s} \le 1
$$
Vedere: [[fusione_componenti]]

### Vincoli di coerenza di flusso di esecuzione
- Per ogni flusso, la successione di componenti attive è prefissa
$$
\forall f \in \mathcal{F} \quad \forall k \in \mathcal{K}  \qquad A^f_{k+1} \le A^f_{k}
$$

- Per ogni flusso, per ogni modello, se un livello è assegnato ad una componente, i suoi predecessori devono essere stati assegnati entro quella componente.
$$
\forall f \in \mathcal{F} \quad \forall m \in \mathcal{M}^f \quad \forall (u,v) \in E_m^f \quad \forall k \in \mathcal{K} \qquad
\sum_{h = 1}^k x^f_{m,v,h} \le \sum_{h = 1}^{k} x^f_{m,u,h}
$$

- Per ogni flusso, per ogni modello, un tensore del modello attraversa il taglio da componente $k$ a componente $k+1$ se e solo la sorgente del tensore è prima del taglio (quindi è assegnata ad una componente in $[1,k]$) e se c'è una destinazione massimale del tensore dopo il taglio (quindi almeno una destinazione è assegnata ad una componente $[k+1,K]$).
$$
\begin{align}
& \forall f \in \mathcal{F} \quad \forall k \in \mathcal{K} \backslash \{K\} \quad \forall m \in \mathcal{M}^f \quad \forall t \in T_m^f 
\\ \\
& \text{if} \; \text{Dst}^{*}(t) = \{v^*\} 
\qquad  c^f_{m,t,k} =\sum_{h=1}^{k}\left(x^f_{m,\text{src}(t),h}-x^f_{m,v^*,h}\right)
\\ \\
&\text{if} \; |\text{Dst}^{*}(t)| > 1 
\qquad 
\begin{cases}
\displaystyle 
c^f_{m,t,k} \ge  \sum_{h=1}^{k} \left( x^f_{m,\text{src}(t),h} - x^f_{m,v,h} \right)
	\qquad \forall v \in \text{Dst}^{*}(t)
\\ \\
\displaystyle
c^f_{m,t,k} \le \sum_{v\in \text{Dst}^{*}(t)} \sum_{h=1}^{k}
	\left( x^f_{m,\text{src}(t),h} - x^f_{m,v,h} \right)
\end{cases}
\end{align}
$$
Nella relazione, le variabili del primo caso possono essere rimosse, visto che sono definite da un'espressione lineare.
## Tempo di risposta
### Tempo di batch fill
Il tempo medio di batch fill è dato da:
$$
T^{f-fill} = \sum_{m \in \mathcal{M^f}} u^f_m \ \frac{B^f_m - 1}{2\Lambda^f}
$$

Inoltre, definiamo il tasso di arrivo riscalato alla dimensione del batch del modello come:
$$
\lambda^f_m = \frac{\Lambda^f}{B^f_m}
$$
Supponiamo comunque che il processo di arrivo si mantenga di Poisson.

### Tempo di esecuzione di una componente
Il tempo di esecuzione di una componente $k$ su un server $s$ è dato da:
$$
T^{f,ex}_{m,k,s} = \sum_{l \in V_m^f} \tau^f_{m,l,s} \cdot x^f_{m,l,k} \cdot e^f_{k,s}
$$

Il tempo di esecuzione di una componente sarà quindi dato da:
$$
T^{f,ex}_{k} = \sum_{m \in \mathcal{M}^f} \sum_{s \in \mathcal{S}^f} T^{f,ex}_{m,k,s}
$$

#### Linearizzazione con upper bound
Scriviamo:
$$
T^{f,ex}_{m,k,s} = 
\sum_{l \in V_m^f} \tau^f_{m,l,s} \cdot x^f_{m,l,k} \cdot e^f_{k,s} = 
e^f_{k,s} \cdot \sum_{l \in V_m^f} \tau^f_{m,l,s} \cdot x^f_{m,l,k} = 
e^f_{k,s} \cdot E^{f,ex}_{m,k,s}
$$
La sommatoria interna è un'espressione affine con upper-bound dato da:
$$
\hat{E}^{f,ex}_{m,s} = \sum_{l \in V_m^f} \tau^f_{m,l,s}
$$

Possiamo linearizzarla con i seguenti vincoli:
$$
\begin{cases}  
0 \le T_{m,k,s}^{f,ex} \le \hat E_{m,s}^f \cdot e_{k,s}^f \\ \\  
T_{m,k,s}^{f,ex} \le E_{m,k,s}^{f,ex} \\ \\
T_{m,k,s}^{f,ex}  \ge E_{m,k,s}^{f,ex} - \hat E_{m,s}^{f,ex} \cdot (1-e_{k,s}^f)  
\end{cases}  
$$
##### Riduzione del vincolo
Poiché questo tempo entra soltanto in direzione peggiorativa nelle parti successive (incluso nel vincolo) possiamo ridurci a:
$$
\begin{cases}  
0 \le T_{m,k,s}^{f,ex} \le \hat E_{m,s}^f \\ \\  
T_{m,k,s}^{f,ex}  \ge E_{m,k,s}^{f,ex} - \hat E_{m,s}^{f,ex} \cdot (1-e_{k,s}^f)  
\end{cases}  
$$
Infatti, un valore più alto del necessario non può mai aiutare, quindi la minimizzazione lo spinge al suo lower bound.

### Tempo di trasmissione di una componente
Definiamo la quantità totale di dati che passa il taglio come:
$$
\Delta^f_{m,k} = \sum_{t \in T^f_m} \delta_t \cdot c^f_{m,t,k}
$$

Possiamo scrivere il tempo necessario alla trasmissione di un unico pacchetto contenente tutti i tensori del contesto, come:
$$
T^{f,tx}_{m,k,s} = \sum_{s^{'} \in \mathcal{S}^f} \left(
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

#### Linearizzazione con upper-bound
Seguendo lo stesso ragionamento del tempo di esecuzione, possiamo scrivere l'espressione del tempo di trasmissione in modo maggiorativo:
$$
\forall s^{'} \neq s \in \mathcal{S}^f \qquad T^{f,tx}_{m,k,s} \ge \frac{1}{\beta_{s s^{'}}}  
\left[ \Delta^f_{m,k} - \hat \Delta_m \cdot (2 - e^f_{k,s} - e^f_{k+1, s^{'}}) \right]
$$
Anche in questo caso, la minimizzazione porta all'abbassamento dei valori e la relazione coincide con il valore vero per la coppia di server selezionata. Questo vale sempre e solo finché le relazioni entrano in modo peggiorativo nel resto del problema.

Definiamo qui $\hat \Delta_m$ come la quantità di dati massima che può passare qualsiasi taglio del modello.
> [!NOTE] Pre-calcolo del $\hat \Delta_{m}$
> Questo valore si può pre-calcolare risolvendo in fase di profiling statico un problema di ottimizzazione ausiliario per la ricerca del max-cut.
> Inoltre, poiché ci serve una maggiorazione, si può trovare anche risolvendo il rilassamento continuo invece del problema ottimo.

Per la latenza possiamo scrivere invece:
$$
\begin{cases}
\xi_{k,s}^f \ge 
\sum_{s^{'} \neq s \; \in \; \mathcal{S}^f} \xi_{s s^{'}} \; (e^f_{k,s} + e^f_{k+1, s^{'}} - 1)
\\ \\
\xi_{k,s}^f \ge  0
\end{cases}
$$
Da qui troviamo la latenza di trasmissione di una componente come:
$$
\xi_{k,s}^f = \sum_{s \in \mathcal{S^f}} \xi^f_{k,s}
$$
### Coda di esecuzione - Prio non preemptive
Indichiamo con $P$ il massimo numero di code di priorità che un server $s$ può gestire (supposto uguale per tutti i server, ma non è un vincolo necessario); supponiamo che le code siano a priorità decrescente ($1 \rightarrow 2 \rightarrow 3 \rightarrow \dots$).

Poiché ogni flusso ha un unico modello assegnato, una classe di richieste assegnata al server è identificata dalla sola coppia $(f,k)$.

Indichiamo con:
- $n_{s,p} = 1$ se e solo se la coda a priorità $p$ è usata su $s$
	- Per come vincolata successivamente, continua in $[0,1]$

Indichiamo con $\mathcal{P}=\{1,\ldots,P\}$ l'insieme degli indici di priorità.

#### Vincoli di coerenza
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
\displaystyle
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}} r^f_{k,s,p}
\\ \\
n_{s,p} \ge r^f_{k,s,p}
\qquad
\forall f \in \mathcal{F}: s \in \mathcal{S}^f,\
\forall k \in \mathcal{K}
\end{cases}
$$

- L'uso delle code di priorità è prefisso sul server:
$$
\forall s \in \mathcal{S},\ \forall p = 1,\ldots,P-1
\qquad
n_{s,p} \ge n_{s,p+1}
$$

#### Formula di coda con variabili
La formula del tempo di coda con priorità non-preemptive in generale è la seguente:
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

Dobbiamo esprimere le variabili $\rho_s^{<p}$ e $\rho_s^{\le p}$  perché il loro valore varia in funzione dell'assegnazione delle classi alle code. Definiamo prima di tutto:
$$
\rho^f_{k,s}
=
\sum_{m \in \mathcal{M}^f}
\lambda^f_m \; T_{m,k,s}^{f,\mathrm{ex}}
$$

Da cui troviamo le espressioni delle utilizzazioni prioritarie:
$$
\rho_s^{<p}
=
\sum_{h=1}^{p-1}
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
r^f_{k,s,h} \; \rho^f_{k,s}
$$
$$
\rho_s^{\le p}
=
\sum_{h=1}^{p}
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
r^f_{k,s,h}\,\rho^f_{k,s}
$$

##### Linearizzazione delle utilizzazioni prioritarie
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
\le \hat \rho^{ex}r^f_{k,s,p}
\qquad \forall p \in \mathcal{P}
\\[8pt]
\displaystyle\sum_{p \in \mathcal{P}}
\bar{\rho}^f_{k,s,p}
=
\rho^f_{k,s}
\end{cases}
$$
Poiché soltanto una priorità può essere selezionata, tutte le altre variabili $\bar{\rho}^f_{k,s,p}$ saranno nulle e quella corrispondente alla priorità attiva assumerà il valore $\rho^f_{k,s}$.

Il valore $\rho^{\max}$ costituisce un upper bound valido per l'utilizzazione di una singola classe, poiché imponiamo:
$$
\rho_s
=
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{k \in \mathcal{K}}
\rho^f_{k,s}
\le \hat \rho^{ex}<1
$$

##### Discretizzazione di $\rho_s^{<p}$
Supponiamo di dividere l'intervallo $[0,\hat \rho^{ex}]$ in $I$ intervalli. Facciamo in modo che $\rho_s^{<p}$ assuma un valore fisso all'interno di questo intervallo; poiché vogliamo maggiorare, ci interessa l'estremo superiore dell'intervallo.

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
\hat{\rho}_I=\hat \rho^{ex}
$$

Definiamo la variabile $g_{s,p,i} \in \{0,1\}$ se e solo se il valore $\rho_s^{<p}$ ricade nell'intervallo $i$-esimo.

Poiché per la prima priorità si ha:
$$
\rho_s^{<1}=0
$$
non è necessario discretizzarne l'utilizzazione cumulativa. La discretizzazione viene quindi applicata soltanto alle priorità $p=2,\ldots,P$.

Di conseguenza, le variabili $g^{ex}_{s,p,i}$ e i relativi vincoli di discretizzazione sono definiti soltanto per:
$$
p=2,\ldots,P,\qquad i=1,\ldots,I
$$

I vincoli di coerenza qui sono i seguenti:
- Data una coda, essa ha esattamente un bin attivo.
  Imponiamo questo vincolo, a prescindere che la coda sia usata o meno, perché anche se una coda non è usata, supposto di porsi in quella coda, comunque il valore di $\rho_s^{<p}$ visto da quella coda è non nullo.
$$
\forall s \in \mathcal{S},\ \forall p=2,\ldots,P
\qquad
\sum_{i=1}^{I}g^{ex}_{s,p,i}=1
$$

- Il valore reale deve cadere nel bin associato
$$
\forall s \in \mathcal{S},\ \forall p=2,\ldots,P
\qquad
\sum_{i=1}^{I}\hat{\rho}_{i-1}g^{ex}_{s,p,i}
\le
\rho_s^{<p}
\le
\sum_{i=1}^{I}\hat{\rho}_i g^{ex}_{s,p,i}
$$

#### Rappresentazione conica
Definiamo $w_{s,p,i}$, che rappresenta il tempo di attesa maggiorato della coda di priorità $p$ quando $\rho^{<p}$ ricade nel bin $i$-esimo :
$$
w_{s,p,i} \ge \frac{H_s}{2(1 - \hat \rho_i) (1 - \rho^{\le p})}
$$

Questa variabile è sempre maggiorata da:
$$
\hat W_{s,i} = \frac{\hat H_s}{2(1 - \hat \rho_i) (1 - \hat \rho^{ex})}
$$
Numeratore, maggiorante di $H_s$:
$$ 
\hat H_s = \min\left\{ \sum_{\substack{f\in\mathcal{F}:\\s\in\mathcal{S}^f}} \max_{m\in\mathcal{M}^f} \left\{ \lambda_m^f \left(\tilde{T}_{m,s}^f\right)^2 \right\} ,\; \hat \rho^{ex} \max_{\substack{f\in\mathcal{F}:\ s\in\mathcal{S}^f\\ m\in\mathcal{M}^f}} \tilde{T}_{m,s}^f \right\} 
$$
Considerando che il vincolo sul tempo di risposta è un requisito stretto:
$$ 
\tilde{T}^f_{m,s} = \max\left\{ 0,\; \min\left\{ \hat{E}^f_{m,s},\; \hat{R}^f-T^{f\text{-}fill}_m \right\} \right\} 
$$

Per la prima priorità, non essendo necessaria la discretizzazione, possiamo rappresentare direttamente il tempo medio di attesa mediante la variabile continua $w_{s,1,0}$, imponendo:
$$
\forall s\in\mathcal{S}
\qquad
\begin{cases}
w_{s,1,0} \ge 0 \\ \\
0\le w_{s,1,0}\le \hat W_{s,0}
\end{cases}
$$

Per le priorità successive $p=2,\ldots,P$, manteniamo invece la rappresentazione discretizzata, imponendo i vincoli:
$$
\forall s \in \mathcal{S},\
\forall p=2,\ldots,P,\
\forall i=1,\ldots,I
\qquad
\begin{cases}
w_{s,p,i}\ge0
\\ \\
w_{s,p,i} \le \hat W_{s,i} \cdot g^{ex}_{s,p,i}
\end{cases}
$$

Per la prima priorità possiamo imporre direttamente il vincolo conico come:
$$2 w_{s,1,0} (1-\rho_s^{\le1}) \ge H_s$$

Per le altre priorità, definiamo le espressioni ausiliarie:
$$
\forall s\in\mathcal{S},\ \forall p=2,\ldots,P \qquad 
\bar w_{s,p}
=
\sum_{i=1}^{I}
(1-\hat{\rho}_i) \; w_{s,p,i}
\qquad
w_{s,p} = \sum_{i=1}^{I} w_{s,p,i}
$$

Dei prodotti interni alla sommatoria, soltanto uno sarà attivo, cioè quello corrispondente al bucket valido per l'utilizzazione.

Imponiamo quindi i vincoli conici ausiliari:
$$
\forall s\in\mathcal{S},\ \forall p=2,\ldots,P
\qquad
2 \; \bar w_{s,p} \; (1-\rho_s^{\le p})\ge H_s
$$
Questa rappresentazione costituisce un vincolo conico ruotato, dato che entrambi i fattori a sinistra sono non negativi e il membro destro è una somma di quadrati pesati.

#### Tempo di coda di esecuzione di una componente
Definiamo quindi il tempo medio di attesa in coda della classe $(f,k)$:
$$
w^{f,\mathrm{ex}}_k
$$

Possiamo imporre:
$$
\begin{cases}
\displaystyle
w^{f,\mathrm{ex}}_k
\ge
w_{s,p} - \hat W_{s,p} \; (1-r^f_{k,s,p})
\qquad
\forall s \in \mathcal{S}^f,\ \forall p \in \mathcal{P}
\\ \\
\displaystyle
w^{f,ex} \ge 0 \\ \\
\displaystyle
w^{f,ex} \le \sum_{s \in \mathcal{S}^f} \sum_{p \in \mathcal{P}} \hat W_{s,p} \; r^f_{k,s,p}
\end{cases}
$$
Con:
$$
\hat W_{s,p} =
\begin{cases}
\displaystyle
\frac{\hat{H}_s}{2 (1 - \hat \rho^{ex})} \qquad p=1
\\ \\
\displaystyle
\frac{\hat{H}_s}{2 (1 - \hat \rho^{ex})^2} \qquad p \ge 2
\end{cases}
$$

Se la classe $(f,k)$ è assegnata alla priorità $p$ sul server $s$, il vincolo diventa:
$$
w^{f,\mathrm{ex}}_k \ge \bar w_{s,p}
$$
L'uguaglianza si ottiene quando il tempo di attesa viene minimizzato o quando i vincoli successivi rendono stretta la maggiorazione.

In questo modo, il tempo di attesa della componente è rappresentato mediante un'unica variabile continua, indipendente dal server e dal modello selezionato.

### Coda di trasmissione - Processor Sharing
Il tempo medio di risposta per la trasmissione del risultato della componente $k$ in Processor Sharing è dato da:
$$
R^{f,tx}_{k} = \sum_{m \in \mathcal{M}^f} \sum_{s \in \mathcal{S}^f}\frac{T^{f,tx}_{m,k,s}}{1 - \rho^{tx}_s}
$$

In cui definiamo il $\rho_{s}^{tx}$ come:
$$
\rho_s^{tx} =
\sum_{\substack{f \in \mathcal{F}:\\s \in \mathcal{S}^f}}
\sum_{m \in \mathcal{M}^f}
\sum_{k \in \mathcal{K}}
\lambda^f_m \cdot T^{f,tx}_{m,k,s}
$$

Imponiamo inoltre il vincolo di stabilità:
$$
\rho_s^{tx} \le \hat \rho^{tx} < 1
\qquad \forall s \in \mathcal{S}
$$

Questa espressione non è direttamente rappresentabile come vincolo MISOCP.
Introduciamo quindi una discretizzazione dell'utilizzazione della rete.

#### Discretizzazione di $\rho_s^{tx}$
Supponiamo di dividere l'intervallo $[0,\hat \rho^{tx}]$ in $I$ intervalli.

Definiamo:
$$
0=\hat{\rho}_0 <
\hat{\rho}_1 <
\hat{\rho}_2 <
\dots <
\hat{\rho}_I=\hat \rho^{tx}
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

#### Linearizzazione con upper-bound
Definiamo quindi la variabile continua non negativa:
$$
R^{f,tx}_k \ge 0
$$

che rappresenta una maggiorazione del tempo medio di risposta della trasmissione relativa alla componente $k$ del flusso $f$.

Sia poi $T^{f,tx}_{k,s} \ge 0$ definita come:
$$
T^{f,tx}_{k,s} = \sum_{m \in \mathcal{M}^f} T^{f,tx}_{m,k,s}
$$

Definiamo la variabile $R^f_{k,s} \ge 0$ con vincoli i vincoli:
$$
\forall i = 1, ... I 
\qquad 
R^f_{k,s} \ge \frac{1}{1 - \hat \rho_i} T^{f,tx}_{k,s} - \hat T^{f,tx}_{k,s} \sum_{j = 1}^{i-1} \left(\frac{1}{1 - \hat \rho_i} - \frac{1}{1 - \hat \rho_j} \right) g^{tx}_{s,j}
$$
Con:
$$
\hat T^{f,tx}_{k,s} = \max_{m \in \mathcal M^f, s^{'} \in \mathcal{S}^f} \frac{\hat \Delta_m}{\beta_{s s^{'}}}
$$

Da cui possiamo imporre:
$$
R^{f,tx}_k \ge \sum_{s \in \mathcal{S}^f} R^{f,tx}_{k,s}
$$


## Modello della memoria
### Memoria di esecuzione della componente su server
La memoria occupata dai pesi e dalle attivazioni sarà data da:
$$
Q^{f,wg}_{m,k,s} = \sum_{l \in V_m^f} \omega^f_{m,l} \cdot x^f_{m,l,k} \cdot e^f_{k,s}
\qquad \qquad
Q^{f,act}_{m,k,s} = \hat \Delta_m \cdot u^f_m \cdot e^f_{k,s}
$$
Questa relazioni suppongono:
- Nel caso dei pesi, che tutti i pesi sino in memoria
- Nel caso delle attivazioni, non sapendo il taglio che verrà fatto, si prende un upper-bound che coincide con la dimensione del taglio massimo.

Per il modello di server che abbiamo considerato, possiamo supporre che un solo modello sia attivo ad un certo istante, ma che tutti i pesi siano caricati (il modello è sempre pronto). Da cui:
$$
Q^{wg}_{s} = \sum_{f \in \mathcal{F}} \sum_{m \in \mathcal{M}^f} \sum_{k \in \mathcal{K}} Q^{f,wg}_{m,k,s}
\qquad \qquad
Q^{act}_{s} = \max_{f,m,k} Q^{f,act}_{m,k,s}
$$

#### Linearizzazione
Nel nostro caso non abbiamo bisogno che il solver trovi il valore esatto della memoria occupata, quindi possiamo accontentarci di maggiorazioni. 

Nel caso dei pesi possiamo imporre:
$$
\begin{cases}
\displaystyle
Q^{f,wg}_{m,k,s} \ge \sum_{l \in V_m^f} \omega^f_{m,l} \cdot x^f_{m,l,k} - \hat Q^{wg}_{m} \cdot (1 - e^{f}_{k,s}) 
\\ \\
\displaystyle
Q^{f,wg}_{m,k,s} \ge 0
\end{cases}
$$

Nel caso delle attivazioni possiamo imporre:
$$
\begin{cases}
Q^{f,act}_{m,k,s} \ge \hat \Delta_m  \cdot (u^f_m + e^f_{k,s} - 1)
\\ \\
Q^{f,act}_{m,k,s} \ge 0
\end{cases}
$$

La memoria complessiva delle attivazioni è definita dal massimo e la possiamo esprimere come:
$$
\begin{cases}
Q^{act}_{s} \ge Q^{f,act}_{m,k,s} 
\qquad \forall f \in \mathcal F, m \in \mathcal M^f, k \in \mathcal{K}
\\ \\
Q^{act}_{s} \ge 0
\end{cases}
$$

### Memoria occupata
Supponendo che la memoria dedicata sia nulla quando il server non ha una GPU, possiamo scrivere quanto segue:
$$
\begin{align}
Q^{wg}_{s} + Q^{act}_{s} & \le M_s^{GPU} \cdot 1_{\{M_s^{GPU} \neq 0\}} + M_s^{RAM} \cdot (1 -  1_{\{M_s^{GPU} \neq 0\}})
\end{align}
$$
Questo significa che se il server ha una GPU, l'unica cosa che pesa sulla memoria principale sono le code, mentre l'esecuzione del modello vero e proprio è su device. In caso contrario, il primo vincolo è banalmente vero e tutto quanto è spostato sulla memoria principale del server.

  
## Requisiti di Qualità
### Tempo medio di risposta
Il tempo medio di risposta per un flusso è dato da:
$$
R^f = T^{f-fill}  + \sum_{k = 1}^{K} \left(w_k^{f,ex-enq} + T_k^{f,ex} \right) + \sum_{k = 1}^{K-1} \left(R^{f,tx}_k + \xi^f_k \right)
$$
Con requisito:
$$
\forall f \in \mathcal{F} \qquad R^f \le \hat{R}^f
$$

## Obiettivo
Indichiamo il tasso di arrivo totale con:
$$\Lambda = \sum_{f \in \mathcal{F}} \Lambda^f$$
Come funzioni obiettivo possiamo considerare la seguente somma pesata sul tasso di arrivo dei flussi
$$
\begin{aligned}
& \sum_{f \in \mathcal{F}} \frac{\Lambda^f}{\Lambda} \ R^f
\\ \\
\end{aligned}
$$

# TODO: Mettere tutto il seguente in unica sezione


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


