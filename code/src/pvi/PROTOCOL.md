# The protocol as the code runs it

This file describes protocol v0 (Sec. `sec:protocol`, `sec:fs`) and its variant V1 (Sec. `sec:v1`) as the library
implements them: the parties, the messages in their order, the bytes on the wire, and the parameters. It is a
reference for checking the implementation against the paper. Every statement names the function that implements
it; where the code and the paper's text differ, the difference is listed at the end. Integers in byte layouts are
little-endian unless stated otherwise; `p = 15 * 2**27 + 1 = 2013265921` is the BabyBear prime
(`pvi.core.field.P`), and `F_{p^8} = F_p[x] / (x^8 - 11)` is the extension field of V1 (`pvi.core.extfield`,
`pvi.core.extfield.D` = 8, `pvi.core.extfield.BETA` = 11). Both parties run in one process; what one party hands the
other is a copy (`pvi.v0.query.passed`), or, with the compact encoding, bytes that the other party decodes.

## 1. The statement

### 1.1 Parties and what each holds

| | Prover | Verifier (client) |
|---|---|---|
| all modes | the integer graph with its weights (`pvi.core.graph.IntGraph`), `pvi.v0.prover.Prover` | the public graph (`pvi.core.graph.IntGraph.public`: the same ops without weights), the parameters (`pvi.v0.params.SecurityParams`), the query `x` |
| mode C | the commitment (`pvi.v0.graph_commitment.GraphCommitment`, from `pvi.v0.graph_commitment.commit_graph`) | `C_M`: per committed matrix a `pvi.commit.commitment.CommitmentPublic` (tag, root, shape, codeword length, members), under a plan the trees (`pvi.commit.commitment.GroupPublic`) and lookup tables (`pvi.commit.commitment.TablePublic`) |
| mode K | the same | the weights `{op: (W, b)}` (`pvi.v0.verifier.Verifier.weights`) |
| mode Kpre | the same | the weights, from which `pvi.v0.verifier.Verifier.precompute` fixes a secret `chi` per weight op checked with Freivalds and its fold `u = chi [W \| b]` once, with an interactive challenger (a Fiat--Shamir one is refused); with `lookups` the verifier reads the embedding rows from its weights at every query (`pvi.v0.verifier.Verifier._own_claim`), and V1 evaluates rule (P4) from them |

`pvi.v0.verifier.Verifier.__post_init__` validates a plan's key (`pvi.v0.verifier.Verifier._check_plan`: every
weight op checked by exactly one committed matrix or table of its shape, every encoded matrix in exactly one group
of its codeword length, a column count for every group). With `lookups=True` (modes K and Kpre only) the prover sends
no claims for the embedding ops (`pvi.v0.verifier.Verifier._sent_ops`).

### 1.2 The graph and its digest

A graph is an ordered list of weight operations (`pvi.core.graph.MatOp`: `Z = W unfold(X) + b`, `W` int8 `[N, K]`,
`b` int64, layout `linear`, `conv` or `embed`) and cheap operations (`pvi.core.graph.CheapOp`), each cheap
operation an `pvi.core.graph.OpFn`: a registered kind with a semantics version (`pvi.core.graph.op_kind`, e.g.
`rope/1`) and its constants. A weight op's claim is `Z` `[N, M]` with `M = pvi.core.graph.MatOp.n_cols(X)`
(positions times batch; for a convolution the unfolded pixels, `pvi.core.graph.MatOp.unfold`); its row length is
`k = K + 1` with a bias, else `K`.

`pvi.core.graph.IntGraph.digest` is the SHA-256 of `b"pvi/graph/v2"` followed by the injective encoding
`pvi.core.graph.canonical` of: for each op in order, a weight op as `["mat", name, inputs, output, n_rows, n_in,
has_bias, layout, conv, max_input]` and a cheap op as `["cheap", name, inputs, output, note, fn, params]` (`fn` by its
kind and every constant: tensors by dtype, shape and bytes, numbers and strings by value, sequences, mappings,
dataclasses, functions and partials element by element); then `[input_name, output_name, meta]` (`meta` holds, for
a generation graph, `lm_positions`). The digest binds every cheap op's integer map and constants and every weight
op's shape, layout and input bound, and not the weights, which the commitment binds. A cheap op whose function is
not an `OpFn` makes the digest raise.

### 1.3 The weight commitment

**Encoding.** A matrix `A = [W | b]` is encoded row by row (`pvi.core.field.rs_encode`): the row is the coefficient
vector of a polynomial of degree below `k`, evaluated at the `n` points `w^c`, `c < n`, with `w =
pvi.core.field.root_of_unity(n) = 31^((p-1)/n)` (`pvi.core.field.GENERATOR` = 31; `n` a power of two up to `2**27`,
`pvi.commit.plans.MAX_N`). So `Enc(row)[c] = sum_j row[j] w^(j c)` (`pvi.commit.commitment.vandermonde_columns`,
`pvi.commit.commitment.codeword_at`).

**Columns and Merkle trees.** Column `c` of the encoded matrix (`N` field elements) is hashed as
`pvi.commit.commitment.column_leaf`: `SHA-256("pvi/col" || tag || c as 8 bytes big-endian || the column as N uint32
little-endian)`, where the tag is the matrix's name. Inner nodes are `SHA-256("pvi/node" || left || right)`
(`pvi.commit.commitment.MerkleTree`, `n` leaves). A multiproof (`pvi.commit.commitment.multiproof`) lists, level by
level from the leaves, in increasing index order, every sibling hash the verifier cannot compute itself, each once;
`pvi.commit.commitment.verify_multiproof` rebuilds the root and requires the proof to be consumed exactly. The
prover keeps no encoding: an opened column is recomputed from the weights as `A V[:, C]`
(`pvi.commit.commitment.WeightCommitment.open`).

**Plans.** `pvi.commit.plans.plan_commitment` maps a policy to a `pvi.commit.plans.CommitmentPlan`:

* `per-op` (no plan, the library default of `commit_graph`): one tree per weight op, `n = rate * next_pow2(k)`
  (`pvi.commit.commitment.WeightCommitment.build`), `rate` = 4 (`experiments.common.cli.RATE`).
* `cnn18c` (the paper's measured configuration, `experiments.common.cli.PAPER`): every encoded matrix at `n =
  max(2**18, 2 next_pow2(m))` for its message length `m`, and the layouts of the suffix `c`:
  * *row*: the matrix `[W | b]` as above;
  * *col*: linear ops that read one tensor with one row length may be stacked and transposed into one matrix
    `[A_1 ; A_2 ; ...]^T` named `q0+k0+v0^T` (`pvi.commit.plans.col_name`): its `K` rows (one per input coordinate,
    of length `sum N_i`) and the biases' row are encoded (`pvi.commit.commitment.TransposedCommitment`);
  * *lookup*: an embedding table without a bias is a Merkle tree over its `V` rows (no code):
    leaf `j` is `SHA-256("pvi/row" || tag || j (8 bytes big-endian) || the d int8 weights of token j)`
    (`pvi.commit.commitment.table_leaf`), padded to `next_pow2(V)` leaves with `SHA-256("pvi/pad" || tag || j)`
    (`pvi.commit.commitment.padding_leaf`) (`pvi.commit.commitment.TableCommitment`).

  The layouts are those of least expected proof bytes at `pvi.commit.plans.REFERENCE_LAMBDA` = 128 (interactive),
  found by descent from the all-row plan (`pvi.commit.plans._choose`); convolutions keep the row layout. Encoded
  matrices of one codeword length share trees (`pvi.commit.plans._runs`: runs of equal-length matrices ordered by
  their exact `t`, split to minimise the expected bytes); the groups are named `g<i>_n<n>`. A group's tree
  (`pvi.commit.commitment.GroupCommitment`) has leaf `c = SHA-256("pvi/group" || group tag || c (8 bytes
  big-endian) || the column digests c of its members, in member order)` (`pvi.commit.commitment.group_leaf`).
  For Llama-2-7B (last block pruned) `cnn18c` gives 130 encoded matrices in six trees of `2**18` leaves and one
  lookup table.

`C_M` is the set of roots (32 bytes each) with the public shapes, tags, codeword lengths and the plan. Before
committing, `commit_graph` refuses any weight op whose honest claims could reach `|z| >= 2**29`
(`pvi.v0.params.claim_bound`: per row `(max_input + 1) sum_j |w_ij| + |b_i|`, for an embedding `max_j |w_ij| +
|b_i|`).

### 1.4 The input

The query `x` is an integer tensor (token ids `[B, T]` for a decoder, images for a CNN). Its shape fixes every claim
column count (`pvi.core.graph.IntGraph.claim_columns`, computed on meta tensors; cached per shape by
`pvi.v0.verifier.Verifier.claim_columns`); a malformed `x` is rejected at `range_or_shape`.

## 2. One query of protocol v0

`pvi.v0.query.run_query(prover, verifier, x, seed=None, wire=False, sampler=None)` runs one query and returns
`{"accepted", "rejected_at", "timings", "contention", "bytes"}`. It checks each message as it arrives and stops at
the first failing check. The challenger is `pvi.v0.transcript.Challenger(fiat_shamir=params.fiat_shamir,
seed=seed)`.

### 2.1 Message flow (mode C)

| Step | Direction | Content | Function |
|---|---|---|---|
| 1 | P -> V | the claims `Z_l` of every sent weight op; under a plan, the lookup tables' rows and one multiproof per table | `pvi.v0.query._claims_message`, `pvi.v0.prover.Prover.claims`, `pvi.v0.prover.Prover.open_tables` |
| - | V | decoding (wire), the token rule, the derivation and range checks, the lookup checks | `pvi.v0.query.decoded_claims`, `pvi.v0.verifier.Verifier.check_tokens`, `pvi.v0.verifier.Verifier.derive`, `pvi.v0.verifier.Verifier.check_lookups` |
| 2 | V -> P | `chi_l` (`r x N_l`) of every row-layout op; `chi'` (`r x M`, or the identity when `M <= r`) of every col matrix | `pvi.v0.verifier.Verifier.fold_challenges` |
| 3 | P -> V | `u_l = chi_l [W_l \| b_l] mod p` (`r x k_l`) of every row-layout op | `pvi.v0.query._fold_message`, `pvi.v0.prover.Prover.fold`, `pvi.commit.commitment.WeightCommitment.fold` |
| - | V | Freivalds' check | `pvi.v0.verifier.Verifier.check_products` |
| 4 | V -> P | `t` distinct column indices of every tree | `pvi.v0.verifier.Verifier.column_challenges` |
| 5 | P -> V | the opened columns of every tree and one multiproof per tree | `pvi.v0.query.open_message`, `pvi.v0.prover.Prover.open` |
| - | V | the column checks | `pvi.v0.verifier.Verifier.column_operands`, `pvi.v0.verifier.Verifier.check_columns` |

In mode K steps 3 to 5 are replaced by the verifier's own fold (`pvi.v0.verifier.Verifier._fold_local`, a fresh
`chi` per query); in mode Kpre `chi` and `u` are the precomputed pair and no challenge is drawn at all.

### 2.2 Challenges

**Interactive.** Each challenge takes a fresh 32-byte key from the operating system (`secrets.token_bytes`, in
`pvi.v0.transcript.Challenger._key`) at the moment it is drawn; with `seed` (tests only) the key is
`SHA-256("test-seed/<seed>/<count>/<label>")`. No key depends on the transcript, and v0 absorbs nothing
interactively.

**Fiat--Shamir.** The transcript is a SHAKE-256 sponge initialised with the 16 bytes `pvi/fullcheck/v3`
(`pvi.v0.transcript.Challenger`). A direct absorption (`pvi.v0.transcript.Challenger.absorb`) appends
`len(label)` (2 bytes big-endian) `|| label || len(message)` (8 bytes big-endian) `|| message`. A large absorption
(`pvi.v0.transcript.Challenger.absorb_large`) is direct below `2**24` bytes; from `2**24` bytes the message is cut
into chunks of `2**22` bytes, leaf `j = SHAKE-256("pvi/leaf/v1" || len(label) (2 bytes) || label || j (8 bytes
big-endian) || chunk_j)` (32 bytes, computed on a thread pool), and the transcript absorbs, under `label + "/tree"`,
the chunk count, the chunk size and the length (three 8-byte big-endian integers) followed by the leaves. The key of a
challenge with label `L` is the first 32 bytes of a copy of the sponge after `"challenge/" || L`; a challenge label
is not absorbed into the transcript itself.

**Expansion** (both settings), from the key `K` (`pvi.v0.transcript.Challenger._stream`: `SHAKE-256(K || name)`):

* field elements (`pvi.v0.transcript.Challenger.folding`, label `fold/<name>`): blocks `fold/<name>/<b>` of 32-bit
  little-endian words; a word below `2p` is accepted and reduced mod `p` (uniform, since `p` divides `2p`); the
  elements fill `chi` `[r, N]` row by row;
* elements of `F_{p^8}` (`pvi.v0.transcript.Challenger.ext`, label `ext/<name>`): the same words in groups of 8
  coefficients (coefficient `i` of `x^i`); with `nonbase`, an element whose coefficients 2 to 7 are all zero is
  drawn again;
* column sets (`pvi.v0.transcript.Challenger.columns`, label `cols/<name>`): 64-bit little-endian words below the
  largest multiple of `n` not above `2**64`, reduced mod `n`, repeats discarded, until `t` distinct indices; the set
  is sorted.

**Absorption order of one v0 query under Fiat--Shamir** (`pvi.v0.query.absorb_statement`, then the messages):

| Label | Content |
|---|---|
| `graph` | `pvi.core.graph.IntGraph.digest` (32 bytes) |
| `params` | `r`, `t`, `rate`, `grinding_bits` as `<IIIi` (under a plan `t` is the largest group's and `rate` is 0), then the mode (`C` or `K`) as ASCII |
| `op/<name>` | per weight op in graph order: `n_rows`, `row_length` (`<QQ`), and if the op has a public of its own, `tag || root || n_points (<Q)` (a group member's root is empty) |
| `col/<name>` | per col matrix: `n_rows`, `row_length`, `n_points`, `len(tag)` (`<QQQH`), `tag`, then each member as `<H` length and name |
| `group/<name>` | per tree of a plan: `len(tag)` (`<H`), `tag`, `root`, `n_points`, `t_g`, member count (`<QQQ`), each member as `<H` length and name |
| `table/<name>` | per lookup table: `len(tag)`, `tag`, `root`, `d`, `V` (`<H`, bytes, 32 bytes, `<QQ`) |
| `lookups` | modes K and Kpre with `lookups`: the names of the embedding ops the verifier reads itself (`<H` length, name) |
| `x` | `pvi.v0.query._tensor_blob(x)`: number of dimensions (`<B`), the shape (`<Q` each), the values as int64 |
| `sampler` | sampled generation only: `pvi.v0.token_sampler.TokenSampler.digest` |
| `claims/PVC3` (large) | wire: the `PVC3` bytes of the claims (section 3.1) |
| `rows/I8` (large) | wire, under a plan with lookup tables: the tables' rows (section 3.3) |
| `claim/<name>` | without wire: per sent claim in sorted name order, its `_tensor_blob` (int32 claims hash as int64) |
| `lookup/<name>` | per lookup table, in graph order: the multiproof's hash count (`<Q`) and its hashes |
| challenges | `fold/<op>` per row-layout op in graph order, then `fold/<col matrix>` per col matrix with `M > r` |
| `u/F31` (large) | mode C, wire: the packed `u` (section 3.2) |
| `u/<name>` | mode C, without wire: per op in sorted name order, `_tensor_blob(u)` |
| challenges | `cols/<op>` per weight op (per-op commitment) or `cols/<group>` per tree, in order |

In mode K the statement and the claims are absorbed and `chi` is read from the transcript; there is no `u` and no
column challenge. In mode Kpre of v0 nothing is absorbed: the proof is one message to a designated verifier whose
`chi` and `u` are its secret key (`pvi.v0.query._claims_message`).

### 2.3 The claims and their checks

The prover runs the graph (`pvi.core.graph.IntGraph.forward`) and keeps every weight op's `Z` `[N, M]`; a lean
prover on a GPU with wire keeps them as int32 (`pvi.v0.claimcodec.narrow`) and encodes them on the device. With
wire, the claims of all sent ops but the lookup tables travel as one `PVC3` frame in graph weight-op order
(`pvi.v0.claimcodec.encode`), the tables' rows as int8 bytes (`pvi.v0.claimcodec.pack_rows`); values the encoding
cannot carry (`pvi.v0.claimcodec.Unencodable`: a claim with `|z| >= 2**29`, a row outside int8) travel as no bytes
(`pvi.v0.query.encoded_message`), which the verifier rejects as malformed.

The verifier, in this order:

1. decodes the claims (`pvi.v0.query.decoded_claims`: `pvi.v0.claimcodec.decode_torch`, or
   `pvi.v0.claimcodec.decode_device` on a GPU, with the shapes `[N, M]` from the public graph and `x`; the tables'
   rows with `pvi.v0.claimcodec.unpack_rows`); a malformed frame is `range_or_shape`;
2. checks the token rule of a generation graph (`pvi.v0.verifier.Verifier.check_tokens`, section 4): `token`;
3. derives every weight op's input (`pvi.v0.verifier.Verifier.derive`, `pvi.v0.verifier.Verifier._derive`): it runs
   the ops in order, recomputing every cheap op from `x` and the claims (recomputation, (S1)); for each weight op it
   requires a claim of shape `[N, M]`, the range check `1 - 2**29 <= z <= 2**29 - 1`
   (`pvi.v0.params.Z_BOUND`), for a lookup table's claim `-128 <= z <= 127`, and for an op that looks rows up ids in
   `[0, V)`; then `pvi.core.graph.MatOp.fold` turns the claim into the op's output tensor. Any failure is
   `range_or_shape`;
4. checks every lookup table (`pvi.v0.verifier.Verifier.check_lookups`): equal ids carry equal columns
   (`lookup_consistency`), and the rows of the distinct ids with the table's multiproof give its root
   (`lookup_merkle`; a proof longer than `ids x depth` hashes is refused).

### 2.4 Freivalds' check (S2)

After the claims, the verifier draws `chi_l` (`[r, N_l]`, uniform over `F_p`) for every op of
`pvi.v0.verifier.Verifier._row_ops` (all weight ops but the col-layout ones, the lookup tables, the embedding ops a
verifier with `lookups` reads itself, and V1's cut ops). The prover sends `u_l = chi_l [W_l | b_l] mod p` `[r, k_l]`
(`pvi.commit.commitment.WeightCommitment.fold`, exact for int8 weights). `pvi.v0.verifier.Verifier.check_products`
checks op by op, in graph order: `u` is int64 of shape `[r, k]` (`pvi.v0.verifier.Verifier._u_ok`), every entry in
`[0, p)`, and `chi Z == u [X ; 1]` mod `p` (`[r, M]` on both sides; `pvi.v0.verifier._rhs`,
`pvi.v0.verifier.Verifier._rhs_all`, `pvi.v0.verifier.Verifier._lhs`). A failure, or a malformed `u` message, is
`freivalds`. Same-shape ops are checked in one batched product; the verdict is that of checking op by op.

### 2.5 The column check (S3)

For every col-layout matrix the verifier computes both sides itself (`pvi.v0.verifier.Verifier.column_operands`):
`w = chi' [X ; 1]^T` `[r, K (+1)]` and `z = chi' Z^T` `[r, sum N]` (`Z`: its ops' claims stacked). It then draws, in
one round after `u`, `t` distinct uniform columns of every tree (`pvi.v0.verifier.Verifier.column_challenges`:
`params.columns` per op under the per-op commitment, `params.columns_for(group)` per group). The prover opens every
tree (`pvi.v0.prover.Prover.open`): each member's encoded columns, stacked in member order (`[sum n_rows, t]`), and
one multiproof. `pvi.v0.verifier.Verifier.check_columns` checks tree by tree:

* shape (`pvi.v0.verifier.Verifier._columns_shape_ok`): int64 `[N, t]` (int32 `[t, N]` with wire), every entry in
  `[0, p)`, at most `t x depth` hashes: else `columns_shape`;
* code, per member (`pvi.v0.verifier.Verifier._code_flags`): `chi . E[:, c] == Enc(u)[c]` for a row matrix, `w .
  E'[:, c] == Enc(z)[c]` for a col matrix, at every opened `c` (`pvi.commit.commitment.codeword_at`): else
  `columns_code`;
* Merkle (`pvi.v0.verifier.Verifier._merkle`): each member's column digests (`pvi.commit.commitment.row_leaves`), a
  group's leaves (`pvi.commit.commitment.group_leaf`), and the multiproof against the root: else `columns_merkle`.

### 2.6 Verdict

`rejected_at` is `None` for an accepted query, else the label of the first failing check: `range_or_shape`,
`token`, `lookup_consistency`, `lookup_merkle`, `freivalds`, `columns_shape`, `columns_code`, `columns_merkle`; V1
adds the labels of section 5.4. With wire, a malformed claims frame is `range_or_shape`, a malformed `u` frame
`freivalds` and a malformed columns frame `columns_shape`, the labels of the same checks without wire.

A verifier with `stream=True` (`pvi.v0.verifier.Verifier.verify_streaming`, for a GPU client) receives every
message first, then runs the same checks in a reordered schedule (each weight op checked as soon as its claim is on
the device, `pvi.v0.streaming.ClaimUploads`) with the same verdicts and labels; a rejected streaming query has also
received, counted and absorbed the messages that `run_query` no longer asks for.

## 3. The wire format and the byte counts

With `run_query(wire=True)` every prover message is a byte string; there is no outer framing (each message's length
is known to the transport, and every decoder rejects trailing or missing bytes). Without wire, the messages are
tensors and the byte counts use 4 bytes per claim (1 per lookup-table entry) and per field element.

### 3.1 Claims: `PVC3` (`pvi.v0.claimcodec`)

```
b"PVC3"  u32 n_ops
per op (graph order):  u32 M  u8 flags (bit 0: centred)
                       [centred: u8 B  i32 lo   -- the row means' segment]
                       u8 B  i32 lo             -- the values' (or residuals') segment
zero bytes up to a multiple of 4
one slot stream per width B in increasing order (widths without values have none)
u32 e  Rice(gaps)  Rice(payloads)               -- the e exceptions
```

Each op is one segment (`B <= 30`, base `lo`), or two when centred: its `N` integer row means `o_i = floor((sum_j
z_ij + M // 2) / M)` and then the residuals `z_ij - o_i`. A value `x` is stored as its slot `(x - lo) mod 2**B`; if
it lies outside `[lo, lo + 2**B)` it is also an exception with high part `h = (x - lo) >> B`. Values are ordered by
width, then segment (a centred op's means before its residuals, ops in graph order), each segment's values row-major
over `[N, M]`. Exception `i` at global position `pos_i` has gap `pos_i - pos_{i-1} - 1` (`pos_{-1} = -1`) and
payload `zigzag(h) - 1`. A slot stream of `n` values of `k` bits (`pvi.v0.claimcodec.pack32`) is `k` rows of `G =
ceil(n / 32)` 32-bit words: value `j G + g` occupies bits `[j k, j k + k)` of the words `g, G + g, 2G + g, ...`
(`4 k G` bytes). A Rice vector (`pvi.v0.claimcodec._rice_encode`) is a byte `k <= 32`, the `k`-bit low parts as a
slot stream, and the quotients `q = v >> k <= 15` in bit-plane unary (level `l = 1..15`: one bit `[q >= l]` per
value still active, little-endian bit order, padded to a byte). The encoder chooses centring and `B` per op from a
sample of rows (at most 256 rows or about `2**16` values) by an estimate of the bits; the host and device encoders
give the bytes of the reference encoder `tests/claimcodec_reference.py`. The decoder (`pvi.v0.claimcodec.decode`)
rejects with `pvi.v0.claimcodec.ClaimCodecError` a bad magic or op count, an `M` other than the expected one (before
allocating), other flag bits, `B > 30`, a base outside `(-2**30, 2**30 - 2**B]`, truncated or trailing bytes,
non-zero padding, `k > 32`, exception positions out of order or range, and values reaching `|x| >= 2**30`.

### 3.2 Field elements: `F31` (`pvi.v0.claimcodec.pack_field`)

The tensors of a message are flattened (row-major) and concatenated, and the elements packed at 31 bits as one slot
stream (`pvi.v0.claimcodec.field_size(n) = 4 * 31 * ceil(n / 32)` bytes); an element outside `[0, 2**31)` is
unencodable. `pvi.v0.claimcodec.unpack_field` requires exactly that length; the verifier's checks require `[0, p)`.

* `u` (step 3): `[u_l for l in _row_ops()]`, each `[r, k_l]`;
* the openings (step 5, `pvi.v0.query.open_message`): per tree in the order of the column challenges, its opened
  columns as `[t, N]` (column after column, each column's `N` entries in member order). The multiproofs travel as
  lists of 32-byte hashes.

### 3.3 Lookup rows: `I8` (`pvi.v0.claimcodec.pack_rows`)

Per lookup table in graph order, its claim `[d, M]` as `M` rows of `d` int8 bytes (position after position);
`pvi.v0.claimcodec.unpack_rows` requires exactly `sum d M` bytes.

### 3.4 In-process formats of the streaming verifier

`pvi.v0.streaming.wire_claim` (int32 claims in pinned host memory), `pvi.v0.streaming.wire_rows` (int8 table rows)
and `pvi.v0.streaming.wire_openings` (int32 rows `[t, N]`, the layout the leaves hash) are tensor formats, not byte
formats; with wire the verifier decodes the byte messages into them.

### 3.5 Byte counts

`run_query` records `out["bytes"]`: `claims` (wire: `len(PVC3) + len(I8)`; else 4 bytes per claim and 1 per table
entry), `u` (wire: `len(F31)`; else `4 r sum k_l`), `columns` (wire: `len(F31)`; else 4 bytes per opened entry) and
`paths` (32 bytes per hash of every table multiproof and tree multiproof, `pvi.v0.query.proof_hashes`). Modes K and
Kpre send only the claims. Not counted: the query `x`, the column indices (derived from the challenges) and any
transport framing. The benchmark records every part as `bytes_<part>` and their sum as `bytes_total`
(`experiments.costs.bench._record_query`); a decoder without a full build is extrapolated linearly from its 1-
and 2-block builds (`experiments/comparison/aggregate.py`). For models too large to run,
`pvi.commit.cost.proof_bytes` gives the same parts from the shapes (claims, `u` and columns exactly; multiproofs as
their expectation, `pvi.commit.cost.expected_multiproof_nodes`, `pvi.commit.cost.expected_lookup_nodes`).

## 4. Generation

**One prefill.** `pvi.models.decoder.with_lm_positions(graph, n)` moves the logits to the last `n` positions
(`meta["lm_positions"] = n`, bound by the digest). A response of `G` tokens to a prompt of `P` tokens is proved as one
query on `x = prompt || y_1 .. y_{G-1}` (`T = P + G - 1` positions) with `n = G`; the prover produces the response
with `pvi.models.decoder.greedy_tokens` or `pvi.v0.token_sampler.sample_tokens`.

**What is checked per token.** `pvi.v0.verifier.Verifier.check_tokens` takes the LM head's claim `[V, B n]` and
requires, for each of the first `n - 1` columns, that its first largest entry (greedy) or the sampler's choice is
the next input token; a query shorter than `n` positions is rejected (`token`). The logits themselves are checked as
every other claim, so a wrong token needs a wrong claim. The last column gives `y_G`.

**The sampler** (`pvi.v0.token_sampler.TokenSampler`, sampled queries need `B = 1`): the client's 32-byte seed, an
integer weight `a = round(c 2**16 / tau)` clamped to `[1, 2**24)`
(`pvi.v0.token_sampler.TokenSampler.for_temperature`, `c` the graph's logit scale) and `top_k >= 0`. The noise at
position `pos` is `G[v] = TABLE[u_v]`, with `u_v` the `v`-th 16-bit little-endian word of
`SHAKE-256("pvi/sampler/v1" || seed || pos as <Q)` and `TABLE[i] = round(2**16 (-ln(-ln((i + 1/2) / 2**16))))`,
checked against its SHA-256 (`pvi.v0.token_sampler.gumbel_table`). The token is the first maximiser of
`a z_v + G[v]` (int64) among the `top_k` largest `z` (ties to the smaller index; all if 0)
(`pvi.v0.token_sampler.TokenSampler.choose`). The token appended at position `P + i` is chosen from the logits and
noise index of position `P + i - 1`. `run_query(sampler=...)` sets the sampler on the verifier, and under
Fiat--Shamir the statement absorbs `sampler` = `"pvi/sampler/v1" || a || top_k (<II) || seed`.

## 5. V1

`pvi.v1.protocol.run_cut_query(prover, verifier, x, seed=None, sampler=None)` runs one V1 query (always in the
compact encoding; modes C and Kpre, non-streaming verifier, one prompt per query). Its parameters come from
`pvi.v0.params.params_for` with `cut=True`, which adds V1's two terms to the union bound (section 7) and sets the
instance bound `cut_lmax`.

### 5.1 The cut

`pvi.v1.cut.CutPlan.from_graph` (via `pvi.v1.protocol.cut_plan`) is a public function of the graph, the commitment
plan (or the verifier's publics) and the query's column counts; prover and verifier compute it independently and a
mismatch raises. An op is cut if (P1) it is a linear weight op, not the graph's output, whose output is read by exactly
one cheap op built by `pvi.core.graph.requant_fn`: a requantisation with clamp in `[-128, 127]`, or a residual
addition reading it as its second input only; (P2) it has all `T` columns; (P3) `floor(2**sh / m) >= 2` and
`ceil(2**sh / m) <= 2**16`; (P4) its claim bound is below `2**29` (evaluated where weights are held; in mode C a
property of the commitment); (P5) each window width is shared by fewer than `2**29` entries (else the op with the most
rows of that width is cleared, repeatedly); members of a col matrix are cut together or not at all. For a cut op the
prover sends `s = clamp(shift(Z), lo, hi)` (requantisation) or `Delta = shift(Z)` (residual), `shift(z) = (m z +
2**(sh-1)) >> sh` (`pvi.v1.cut.requant_values`); the window of value `v` is `[lo(v), lo(v) + W(v))` with `lo(v) =
ceil((v 2**sh - 2**(sh-1)) / m)` and `W(v) = lo(v + 1) - lo(v)` (`pvi.v1.cut.lo`, `pvi.v1.cut.windows`); a clamped
entry outside its window is an exception with its exact `z` and window `[z, z + 1)` (`pvi.v1.cut.split`).

**Instances.** Units in graph order (a cut col matrix with all its members, every other cut op alone) are packed
greedily into instances of at most `2**lmax` entries (`lmax = params.cut_lmax`, default `pvi.v1.cut.LMAX` = 27). An
instance stacks its ops by rows; entry `(row, col)` is leaf `row 2**n_col + col` with `n_col = ceil(log2 T)` and
`n_row = ceil(log2 rows)` (`pvi.v1.cut.CutPlan.n_vars`). The table (`pvi.v1.logup.table`) is `{t + x w : w in
widths, 0 <= t < w}`, widths ascending including 1. `pvi.v1.cut.CutPlan.digest` is the SHA-256 of `pvi/cut/v1` and
the canonical encoding of the field's parameters, `lmax`, `T`, the widths, every cut op with its placement and the
instances.

### 5.2 Messages and challenges, in order (`pvi.v1.protocol._query`)

| Step | Direction | Content | Encoding |
|---|---|---|---|
| 1 | P -> V | the values frame: the cut ops' `s`, the clear ops' claims `Z` (graph order), and the lookup rows and multiproofs | `PVC3`, `I8` (sections 3.1, 3.3) |
| 1 | P -> V | the exceptions | `PVX1` (`pvi.v1.protocol.pack_exceptions`): `b"PVX1"`, per cut op (plan order) its count (u32), then per op its flat row-major indices (u32, ascending) and exact claims (i32) |
| 1 | P -> V | the multiplicities `m_t` | `PVC3` of one op `[1, |table|]` |
| 2 | V -> P | `alpha` in `F_{p^8}` outside `F_p + x F_p` (`ext/cut/alpha`, `nonbase`); in mode C v0's `chi` of the clear row-layout ops and clear col matrices | |
| 3 | P <-> V | per instance `b`, the fractional-sum GKR (`pvi.v1.gkr.prove`, or `pvi.v1.gkr_triton` with the same bytes) | `u8 n` then `F31` of the `8 n_elements(n)` coefficients (`pvi.v1.gkr.Transcript.to_bytes`, `pvi.v1.gkr.n_elements(n) = 4 + sum_{k<n} (3k + 4)`) |
| 4 | P -> V | the folds: in mode C v0's `u` of the clear row ops, then per fold item (`pvi.v1.protocol._fold_items`): a cut row-layout op's `u = chi^T A` as 8 base-field planes `[8, k]`, a cut col matrix's `y = A xbar` `[8, sum N]`; in Kpre every cut op's `y` `[8, N]` | one `F31` frame |
| 5 | V -> P | mode C: the column indices of every tree | |
| 6 | P -> V | mode C: the openings | as v0 |

Under Fiat--Shamir the transcript absorbs the statement (`pvi.v0.query.absorb_statement`, also in Kpre), `cut`
(the plan digest), `claims/PVC3`, `rows/I8`, `lookup/<name>`, `cut/exc` and `cut/mult` (large), before `alpha` and
`chi`. The GKR of instance `b` absorbs `cut/<b>/top` (the root's four children), draws `cut/<b>/mu0`, and for each
layer `k = 1 .. n-1` draws `cut/<b>/lam<k>`, then per round `j` absorbs `cut/<b>/g<k>.<j>` (`g(0), g(2), g(3)`)
and draws `cut/<b>/r<k>.<j>`, then absorbs `cut/<b>/v<k>` (the four child values) and draws `cut/<b>/mu<k>`; these
messages are absorbed as the decoded coefficients (int64, `pvi.v1.gkr._absorb`). The prover runs the GKR on a fork
of the transcript (`pvi.v0.transcript.Challenger.fork`); interactively the verifier draws each round challenge when
the prover asks for it and replays the same values (`pvi.v1.protocol._Recorder`, `pvi.v1.protocol._Replayer`). The
fold frame is absorbed as `u/F31` before the columns (mode C, `pvi.v0.query.open_message`), and in Kpre after the
last challenge.

### 5.3 The logUp-GKR

A cut entry with residue `delta = Z - L` and width `W` is the leaf `(1, alpha - (delta + x W))`, padding leaves
`(0, 1)` (`pvi.v1.logup.leaves`). The fraction tree combines children into `(p_L q_R + p_R q_L, q_L q_R)`
(`pvi.v1.gkr.fraction_layers`). The prover sends the root's two children, then per layer a degree-3 sum-check over
`P + lambda Q` binding the point's bits from the least significant one, and the four child values; the verifier
(`pvi.v1.gkr.verify`, or batched over instances, `pvi.v1.gkr.verify_batch`, with the same verdict) checks every
round with `g(1) = c - g(0)` and returns the root `(p0, q0)`, the point `rho` (its first `n_col` coordinates
`r_col`) and the claims `(P_hat, Q_hat)` on the leaf layer.

### 5.4 The verifier's checks, in order, and their labels

1. decoding of the values frame: `range_or_shape`;
2. the token rule (`pvi.v0.verifier.Verifier.check_tokens`): `token`;
3. the derivation (`pvi.v0.verifier.Verifier._derive` with the cut plan): every cut op's values in its range (a
   requantisation's clamp, a residual's `R_Delta = [shift(1 - 2**29), shift(2**29 - 1)]`, `pvi.v1.cut.delta_range`),
   its input within `+-(max_input + 1)`, clear claims as in v0, the consumer of a cut op reading `s`: `range_or_shape`;
4. the lookup checks: as in v0;
5. (D4) exceptions (`pvi.v1.protocol.unpack_exceptions`, `pvi.v1.protocol.exceptions_ok`): requantisations only,
   strictly increasing indices inside the op, `|z| < 2**29`, `rq(z)` equal to the sent value: `cut_exception`;
6. (D5) multiplicities (`pvi.v1.logup.multiplicities_ok`): `|table|` counts, each `0 <= m_t < p`, summing to the
   number of leaves as integers: `cut_multiplicities`;
7. (G) every GKR instance (`pvi.v1.logup.verify_instances`): parsed, of the plan's size, every round, `q0 != 0`:
   `cut_gkr`;
8. the fold frame: malformed is `freivalds` in mode C with clear row ops, else `cut_final`; a cut fold outside
   `[0, p)` is `cut_final`;
9. (F1) Freivalds' check of the clear ops (`pvi.v0.verifier.Verifier.check_products`): `freivalds`;
10. (F2) Kpre: `chi_s^T y = u_s xbar` in `F_{p^8}` for every cut op and secret vector (`pvi.v1.protocol._kpre_y_ok`):
    `kpre_y`;
11. (F3) per instance, `P_hat = I(rho)` and `Q_hat = alpha I - sum_l (S_l - chi.(L e) + x chi.(W e)) + (1 - I)`, with
    `e = eq(r_col)[0:T]`, `chi = eq(r_row)` at the op's rows, `S_l = u_l . xbar_l` (row layout, mode C) or `chi . y_l`
    (`pvi.v1.protocol._final_ok`, `pvi.v1.protocol._final_claims`, `pvi.v1.logup.indicator`, `pvi.v1.products`):
    `cut_final`;
12. (F4) `sum_b p0/q0 == sum_t m_t / (alpha - tau_t)` (`pvi.v1.logup.logup_ok`; by batch inversion with the default
    optimised verifier, by cross-multiplication with `v1_verifier="baseline"`): `cut_logup`;
13. mode C: the column checks, with the 8 planes of `chi` and `u` (cut row layout), of `xbar` and `y` (cut col
    matrix), or v0's operands: the v0 labels.

`out["bytes"]` adds `cut_exc`, `cut_mult` and `cut_gkr` to v0's parts; `pvi.v1.protocol.cut_proof_bytes` gives every
part without running the GKR, the folds or the openings.

## 6. The setup proof and the setup cache

**Setup proof** (`pvi.commit.setup_proof`, for the trees of a commitment plan; always non-interactive). Its
transcript is a separate Fiat--Shamir `pvi.v0.transcript.Challenger` (`pvi.commit.setup_proof._transcript`): it
absorbs `setup/params` (`s`, `lambda` as float64, the grinding bits: `<IdI`), `setup/graph` (the graph digest) and
per group `setup/group/<g>` (tag, root, `n`, `t0`, member count, and per member its tag, `n_rows`, `row_length`,
`n_points` and radius `e`). Then:

1. the verifier draws `psi_j` (`s x R_j`, labels `fold/setup/<m>`) for every member matrix;
2. the committer sends `psi_j A*_j` (`s x m_j`, `pvi.commit.setup_proof._combine`), absorbed as `setup/msg/<m>`
   (shape `<II` and int64 values);
3. the verifier draws `t0_g` columns per tree (`cols/setup/<g>`); the committer opens them with one multiproof
   (`pvi.commit.setup_proof.prove_setup`);
4. `pvi.commit.setup_proof.verify_setup` checks the parameters (`pvi.commit.setup_proof.params_sound`), every message
   (int64, shape `(s, m_j)`, in `[0, p)`), every opening (the indices equal the derived ones, int64, shape, in
   `[0, p)`), per member `Enc(psi_j A*_j)[c] == psi_j E_j[:, c]`, and the multiproof of the recomputed group leaves.

After an accepted setup proof, queries open the per-query `t` of `pvi.commit.setup_proof.setup_params` (computed with
`k + e` in place of `k`), installed by `pvi.commit.setup_proof.after_setup`. Its size
(`pvi.commit.setup_proof.SetupProof.nbytes`, `experiments.costs.setup_bytes`) counts 4 bytes per field element and
32 per hash; the code defines no byte encoding of it.

**Setup cache** (`pvi.commit.setup_cache`, opt-in: `commit_graph(cache=DIR)`). It stores the prover's leaf digests
of every tree (`op:<name>`, `group:<g>`, `table:<t>`) in `<DIR>/<key>.npz`; the key
(`pvi.commit.setup_cache.key`) is a BLAKE2b-256 over a format version, the base rate, the plan (policy, groups and
members, every matrix's layout, shape and codeword length) and every weight op's name, shape and BLAKE2b-256 of its
int8 weights and int64 bias. On reading, `commit_graph` recomputes `cache_checks` (default 4) random leaves of every
tree from the weights (`pvi.commit.setup_cache.check_group`, `pvi.commit.setup_cache.check_matrix`,
`pvi.commit.setup_cache.check_table`) and compares them; a mismatch, a missing tree or an unreadable file is a miss,
and the commitment is computed afresh and stored. The result (roots, openings, proofs) is that of a fresh
commitment; the verifier is unaffected, since it recomputes the leaves of every opened column.

## 7. The soundness parameters

| Quantity | Rule in the code | Function |
|---|---|---|
| field | `p = 15 * 2**27 + 1`, `log2 p ~ 30.907`; NTT lengths up to `2**27` | `pvi.core.field.P`, `pvi.core.field.LOG2_P`, `pvi.core.field.TWO_ADICITY` |
| `n_checks` | the number of weight ops of the whole model (each member of a col matrix and each embedding counted), passed by the scripts (e.g. `len(decoder_shapes(cfg))`, 226 for Llama-2-7B) | `pvi.commit.cost.decoder_shapes`, `experiments/costs/bench.py:806-809` |
| `beta` | `lambda + log2(2 max(1, n_checks))`, plus `grinding_bits` = 64 under Fiat--Shamir; V1 adds 2 to `n_checks` | `pvi.commit.plans.column_bits`, `pvi.v0.params.params_for` |
| `r` | `max(1, ceil(beta / log2 p))` (Llama-2-7B at `lambda` = 128: 5 interactive, 7 Fiat--Shamir) | `pvi.v0.params.params_for` |
| `t`, per-op commitment | `max(1, ceil(beta / log2 rate))` for every op | `pvi.v0.params.params_for` |
| `t_g`, plan | the largest over the group's members of the least `t` with `prod_{i<t} (m-1-i)/(n-i) <= 2**-beta` (Llama-2-7B, `cnn18c`: 23 to 46 interactive, 34 to 67 Fiat--Shamir) | `pvi.commit.plans.CommitmentPlan.group_columns`, `pvi.commit.plans.exact_columns`, `pvi.commit.plans.column_error_log2` |
| plan layouts and groups | chosen once at `lambda` = 128, interactive, on the whole model | `pvi.commit.plans.REFERENCE_LAMBDA`, `pvi.commit.plans.plan_commitment` |
| reported bound | `-log2 sum_j (p**-r + column term)`, less the grinding bits under Fiat--Shamir | `pvi.v0.params.soundness_bits` |
| claim range | `|z| < 2**29` | `pvi.v0.params.Z_BOUND`, `pvi.v0.params.claim_bound` |
| setup proof | `beta0 = lambda + 64 + log2(2 J)` (`J` coded matrices); `s = ceil((beta0 + log2 max n) / log2 p)`; radius `e = floor(f (n - m) / 2)`, `f` in `2**0 .. 2**-10` minimising `t0 / amortise + t`; `t0` least with `prod (n-e-1-i)/(n-i) <= 2**-beta0`; per-query `t` least with `prod (m+e-1-i)/(n-i) <= 2**-query_bits` | `pvi.commit.setup_proof.setup_params`, `pvi.commit.setup_proof.survivors`, `pvi.commit.setup_proof.FRACTIONS`, `pvi.commit.setup_proof.params_sound` |
| V1 | `F_{p^8}`, `alpha` outside `F_p + x F_p`; instances of at most `2**27` leaves | `pvi.core.extfield`, `pvi.v1.cut.LMAX`, `pvi.v0.params.SecurityParams` |
| transcript | SHAKE-256, 32-byte keys and leaves; SHA-256 Merkle trees with 32-byte roots | `pvi.v0.transcript.Challenger`, `pvi.commit.commitment.HASH_BYTES` |

## Differences from the paper's description

Each item gives the paper's text (file and line in `paper_sp2027/`) and what the code does (file and line under
`code/`); which side changes is left to a decision.

1. `paper_sp2027/sections/defence.tex:170` sets `beta = lambda + log2(2|J|)` over the committed matrices; the code
   uses the number of weight operations `n_checks >= |J|` (`src/pvi/v0/params.py:60`, called with the model's op count
   at `experiments/costs/bench.py:806-809`), as the body of `tab:params` (`paper_sp2027/fallback/tab_params.tex:4-5`)
   and `paper_sp2027/sections/ext_proofs.tex:720` state.
2. `paper_sp2027/sections/defence.tex:170-172` says each tree opens the least `t` meeting the column term; this holds
   under a plan, while the per-op commitment opens `ceil(beta / log2 rate)` columns (`src/pvi/v0/params.py:63-65`).
3. `paper_sp2027/sections/defence.tex:49-51` says a Kpre verifier holds only the secret vectors and folded rows; with
   `lookups` (the paper's configuration) it also reads the embedding tables at every query
   (`src/pvi/v0/verifier.py:413-418`), and V1 evaluates (P4) from its weights (`src/pvi/v1/protocol.py:222-223`, as
   `paper_sp2027/sections/ext_proofs.tex:754` says).
4. `paper_sp2027/sections/appendix_v1.tex:63-68` lists (D1), (D2), then (D3) the token rule; the code checks the token
   rule after decoding and before the derivation and its range checks (`src/pvi/v1/protocol.py:369-387`; v0 alike,
   `src/pvi/v0/query.py:372-401`), so a query failing both reports `token`.
5. `paper_sp2027/sections/appendix_v1.tex:82-83` says (F4) is compared by cross-multiplication; the default
   (optimised) verifier forms both sums by batch inversion (`src/pvi/v1/logup.py:111-117`), the baseline one
   cross-multiplies (`src/pvi/v1/logup.py:118-120`); the two tests agree when every denominator is non-zero.
6. `paper_sp2027/sections/v1.tex:143-147` lists the first message, then `alpha`, then the GKR; in mode C the code also
   draws v0's `chi` of the clear operations with `alpha`, before the GKR (`src/pvi/v1/protocol.py:408-410`), and the
   fold message carries the clear `u` with the cut folds (`src/pvi/v1/protocol.py:464-469`).
7. `paper_sp2027/sections/ext_proofs.tex:398-406` lists "the implementation's" labels; it omits the setup proof's
   transcript (`setup/params`, `setup/graph`, `setup/group/<g>`, `setup/msg/<m>`, challenges `fold/setup/<m>` and
   `cols/setup/<g>`; `src/pvi/commit/setup_proof.py:147-173`), which shares the domain string `pvi/fullcheck/v3` and
   starts with `setup/params`, not `graph` (relevant to the `sigma_0` argument at
   `paper_sp2027/sections/ext_proofs.tex:458-461`).
8. Code names in `paper_sp2027/sections/ext_proofs.tex` that no longer exist: "code/src/pvi/fullcheck" (line 43; now
   `code/src/pvi`), `transformer.build_decoder` (93; `pvi.models.decoder.build_decoder`), `protocol.claim_bound`
   (196; `pvi.v0.params.claim_bound`), `real_weights.build_opt_from_hf` and `real_llama.build_llama_from_hf` (224-225;
   `pvi.models.hf_opt.build_opt_from_hf`, `pvi.models.hf_llama.build_llama_from_hf`), `_absorb_statement` (358;
   `pvi.v0.query.absorb_statement`), `cut_protocol.*` (561, 573, 583, 637, 694, 750, 928; `pvi.v1.protocol`),
   `_field_message` (572; `pvi.v0.query.field_message`), `logup_gkr.Transcript.from_bytes` (574;
   `pvi.v1.gkr.Transcript.from_bytes`), `_decoded_claims` (582; `pvi.v0.query.decoded_claims`), `_open_message`
   (583, 694; `pvi.v0.query.open_message`), `protocol._disagrees` (927; `pvi.v0.verifier._disagrees`); and paths to
   development notes outside the artifact (`docs/improvements/...`, lines 656 and 995).
