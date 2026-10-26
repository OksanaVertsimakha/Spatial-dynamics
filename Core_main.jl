###############################################################################
#  Model_core_10_26.jl  —  gene-drive population model (core functions)
#
#  Reorganised for Julia 1.12 / 1.13.
###############################################################################

using Random
using Distributions
using ProgressMeter
using LinearAlgebra
using SparseArrays
using JLD2
using Distributed                      # pmap / CachingPool in the Sp_run* drivers
using PrettyTables                     # only for show=true tables
import StatsBase: ecdf, sample, countmap
import GeometryBasics                  # GeometryBasics.Point2f in Hexagon1 / Hexagon11
import GeometryBasics: Polygon, Point


#==============================================================================
  0. THREADING HELPERS
  Work is split into at most nthreads(:default) contiguous chunks; each chunk
  runs as one task with its own buffers. With 1 thread → plain serial loop.
==============================================================================#

"Split 1:n into at most `nthreads(:default)` contiguous ranges."
function _chunks(n::Integer)
    k = min(n, Threads.nthreads(:default))
    k <= 1 && return [1:n]
    return collect(Iterators.partition(1:n, cld(n, k)))
end

"Run `f(range)` for each chunk of 1:n, in parallel when >1 thread is available."
function _tforeach_chunk(f::F, n::Integer) where {F}
    chunks = _chunks(n)
    if length(chunks) == 1
        f(chunks[1])
    else
        @sync for r in chunks
            Threads.@spawn f(r)
        end
    end
    return nothing
end


#==============================================================================
  1. LIFE-CYCLE STEPS (single site)
  Public keyword functions keep their original signatures; the work is done
  in typed positional kernels (`::Type{T}`) so the compiler specialises on tp.
==============================================================================#

# Draw offspring of `n` fertile pairs with gamete-genotype probs `probs`, then
# split each genotype into females (prob q) and males. Adds into N_exp_f/m.
@inline function _offspring_pair!(N_exp_f, N_exp_m, n::Integer, probs, q::Real, buf)
    rand!(Multinomial(n, probs), buf)
    @inbounds for k in eachindex(buf)
        nk = buf[k]
        if nk > 0                       # Binomial(0, q) is always 0 → skip draw
            s = rand(Binomial(nk, q))
            N_exp_f[k] += s
            N_exp_m[k] += nk - s
        end
    end
    return nothing
end

function _mating(Nf, Nm, E, ff, r_sex, beta, wa, M2, ::Type{T}) where {T}
    N_exp_f = zeros(T, M2)
    N_exp_m = zeros(T, M2)

    Nm_w   = Nm .* wa                   # mating-weighted males
    Nm_sum = sum(Nm_w)
    Nm_sum == 0 && return N_exp_f, N_exp_m          # no males → no offspring
    pAm    = Nm_w ./ Nm_sum                         # male genotype choice probs
    p_beta = Nm_sum / (beta + Nm_sum)               # prob. a female finds a mate

    # Mating pairs: row i = female genotype, column j = male genotype
    N_pairs_A = zeros(T, M2, M2)
    @inbounds for i in 1:M2
        if Nf[i] > 0
            n_pairs = round(T, Nf[i] * p_beta)
            n_pairs > 0 && rand!(Multinomial(n_pairs, pAm), view(N_pairs_A, i, :))
        end
    end

    # Fertile pairs (Poisson on fecundity ff) → offspring genotypes → sex
    buf = zeros(T, M2)
    @inbounds for j in 1:M2, i in 1:M2
        a = N_pairs_A[i, j]
        a > 0 || continue
        n_fert = rand(Poisson(a * ff[i, j]))
        n_fert > 0 && _offspring_pair!(N_exp_f, N_exp_m, n_fert, E[i][j], 1 - r_sex[j, i], buf)
    end
    return N_exp_f, N_exp_m
end

# Mating model (keyword API unchanged)
Mating(; Nf, Nm, E, ff, r_sex, beta=0, M2=10, M=4, tp=Int64, wa=1) =
    _mating(Nf, Nm, E, ff, r_sex, beta, wa, M2, tp)

# Viability selection: each individual survives with prob wf[i] / wm[i]
function _fitness_change(Nf, Nm, wf, wm, M2, ::Type{T}) where {T}
    Nf_new = Vector{T}(undef, M2)
    Nm_new = Vector{T}(undef, M2)
    @inbounds for i in 1:M2
        Nf_new[i] = rand(Binomial(Nf[i], wf[i]))
        Nm_new[i] = rand(Binomial(Nm[i], wm[i]))
    end
    return Nf_new, Nm_new
end

Fitness_change(; Nf, Nm, wf, wm, M2=10, tp=Int64) = _fitness_change(Nf, Nm, wf, wm, M2, tp)

# Density-dependence scale parameter alpha from carrying capacity N and fecundity f
function Alpha(; N, f, phi1=1, phi2=1)
    if phi2 == 1
        a = (N * f / 2) / (phi1 * f / 2 - 1)
    else
        a = (N * f / 2) / (phi1 * f / 2 - 1)^(1 / phi2)
    end
    return a
end

# Density-dependent survival p = phi1 / ((Nh/alpha)^phi2 + 1), same p for all genotypes
function _ddm_change(Nf, Nm, Nh, alpha, phi1, phi2, M2, ::Type{T}) where {T}
    p = phi1 / ((Nh / alpha)^phi2 + 1)
    Nf_new = Vector{T}(undef, M2)
    Nm_new = Vector{T}(undef, M2)
    @inbounds for i in 1:M2
        Nf_new[i] = rand(Binomial(Nf[i], p))
        Nm_new[i] = rand(Binomial(Nm[i], p))
    end
    return Nf_new, Nm_new
end

Ddm_change(; Nf, Nm, Nh, alpha, M2=10, tp=Int64, phi1=1, phi2=1) =
    _ddm_change(Nf, Nm, Nh, alpha, phi1, phi2, M2, tp)


#==============================================================================
  2. ONE-GENERATION MODELS (single site)
    Model2: mating → density dependence → fitness
    Model1: mating → fitness → density dependence
==============================================================================#

function Model2(; Nf, Nm, E, wm, wf, alpha=10^6, ff, r_sex=0.5, beta=1, M2=10, M=4,
                tp=Int64, phi1=1, phi2=1, wa=1)
    Nf0, Nm0 = _mating(Nf, Nm, E, ff, r_sex, beta, wa, M2, tp)
    Nh = sum(Nf0) + sum(Nm0)
    Nf1, Nm1 = _ddm_change(Nf0, Nm0, Nh, alpha, phi1, phi2, M2, tp)
    Nf2, Nm2 = _fitness_change(Nf1, Nm1, wf, wm, M2, tp)
    return [Nf2, Nm2]
end

function Model1(; Nf, Nm, E, wm, wf, alpha, ff, r_sex=0.5, beta=1, M2=10, M=4,
                tp=Int64, phi1=1, phi2=1, wa=1)
    Nf0, Nm0 = _mating(Nf, Nm, E, ff, r_sex, beta, wa, M2, tp)
    Nf1, Nm1 = _fitness_change(Nf0, Nm0, wf, wm, M2, tp)
    Nh = sum(Nf1) + sum(Nm1)
    Nf2, Nm2 = _ddm_change(Nf1, Nm1, Nh, alpha, phi1, phi2, M2, tp)
    return [Nf2, Nm2]
end


#==============================================================================
  3. SHARED SET-UP HELPERS
==============================================================================#

# Genotype rows carrying the sex-distorting drive, by number of alleles M
function _drive_rows(M)
    M == 4 && return (2, 5, 6, 7)
    M == 5 && return (2, 6, 7, 8, 9)
    M == 6 && return (2, 7, 8, 9, 10)
    return ()
end

# Sex-ratio matrix r_sex[male genotype, female genotype].
# A scalar mD_sex fills the drive rows; a matrix is used as given.
function _make_r_sex(M, M2, mD_sex)
    length(mD_sex) >= 2 && return mD_sex
    r_sex = fill(0.5, M2, M2)
    for i in _drive_rows(M)
        r_sex[i, :] .= mD_sex
    end
    return r_sex
end

# Per-site alpha vector. alpha == 0 → from each site's initial population.
function _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)
    Nsites = size(Ntf_init, 2)
    if alpha isa Number
        if alpha == 0
            Npop = vec(sum(Ntf_init .+ Ntm_init; dims=1))   # FIX: per site, was a scalar
            return [Alpha(; N=Npop[s], f=f, phi1=phi1, phi2=phi2) for s in 1:Nsites]
        end
        return fill(float(alpha), Nsites)                 # scalar alpha → every site
    end
    return alpha
end


#==============================================================================
  4. MIGRATION
  Migration from a source site uses column `source` of the adjacency matrix A.
==============================================================================#

# Column-wise neighbour lists of A (dense or sparse).
# normalize=true rescales each column to sum to 1 (used by Migration_inter).
function adjacency_lists(A::AbstractMatrix; normalize::Bool=false)
    S = A isa SparseMatrixCSC ? A : sparse(A)
    N = size(S, 2)
    neighbors = Vector{Vector{Int}}(undef, N)
    probs     = Vector{Vector{Float64}}(undef, N)
    rows = rowvals(S)
    vals = nonzeros(S)
    for col in 1:N
        rng = nzrange(S, col)
        neighbors[col] = rows[rng]
        p = Float64.(vals[rng])
        if normalize && !isempty(p)
            p ./= sum(p)
        end
        probs[col] = p
    end
    return neighbors, probs
end

# Move individuals from `sources` into Nf_out / Nm_out (adds; does not zero).
function _migrate_range!(Nf_out, Nm_out, Nf_s, Nm_s, neighbors, probs, w_scaled,
                         sources, ::Type{T}) where {T}
    M2   = size(Nf_s, 1)
    buf  = Vector{T}(undef, maximum(length, neighbors; init=0))
    @inbounds for source in sources
        neigh = neighbors[source]
        k = length(neigh)
        k == 0 && continue
        p   = probs[source]
        mig = view(buf, 1:k)
        ws  = w_scaled[source]
        for g in 1:M2
            nm = round(T, Nm_s[g, source] * ws)          # males
            if nm > 0
                rand!(Multinomial(nm, p), mig)
                for q in 1:k
                    Nm_out[g, neigh[q]] += mig[q]
                end
            end
            nf = round(T, Nf_s[g, source] * ws)          # females
            if nf > 0
                rand!(Multinomial(nf, p), mig)
                for q in 1:k
                    Nf_out[g, neigh[q]] += mig[q]
                end
            end
        end
    end
    return nothing
end

# Threaded migration: each chunk of source sites gets private output arrays,
# which are summed at the end (no shared writes, no threadid()).
function _migration(Nf_s, Nm_s, neighbors, probs, w_scaled, ::Type{T}) where {T}
    M2, N  = size(Nf_s, 1), length(neighbors)
    chunks = _chunks(N)
    if length(chunks) == 1
        Nf_new = zeros(T, M2, N)
        Nm_new = zeros(T, M2, N)
        _migrate_range!(Nf_new, Nm_new, Nf_s, Nm_s, neighbors, probs, w_scaled, 1:N, T)
        return Nf_new, Nm_new
    end
    tasks = map(chunks) do r
        Threads.@spawn begin
            Nf_loc = zeros(T, M2, N)
            Nm_loc = zeros(T, M2, N)
            _migrate_range!(Nf_loc, Nm_loc, Nf_s, Nm_s, neighbors, probs, w_scaled, r, T)
            (Nf_loc, Nm_loc)
        end
    end
    Nf_new, Nm_new = fetch(tasks[1])::Tuple{Matrix{T},Matrix{T}}
    for t in 2:length(tasks)
        Nf_t, Nm_t = fetch(tasks[t])::Tuple{Matrix{T},Matrix{T}}
        Nf_new .+= Nf_t
        Nm_new .+= Nm_t
    end
    return Nf_new, Nm_new
end

# Adult migration. Pass (neighbors, probs) from adjacency_lists (fast), or A
# (lists are then rebuilt on every call, as the older callers expect).
function Migration_change(; Nf_s, Nm_s, neighbors=nothing, probs=nothing, A=nothing,
                          w, mu_mig=1.0, M2=10, tp=Int64, Nsites=nothing)
    if neighbors === nothing
        A === nothing && throw(ArgumentError("Migration_change needs (neighbors, probs) or A"))
        nb, pr = adjacency_lists(A)
        return _migration(Nf_s, Nm_s, nb, pr, mu_mig .* w, tp)
    end
    return _migration(Nf_s, Nm_s, neighbors, probs, mu_mig .* w, tp)
end

# Old dense-matrix interface, kept for compatibility (same distribution).
Migration_change_old(; Nf_s, Nm_s, A, w, Nsites=size(A, 1), mu_mig=1.0, M2=10, tp=Int64) =
    Migration_change(; Nf_s=Nf_s, Nm_s=Nm_s, A=A, w=w, mu_mig=mu_mig, M2=M2, tp=tp)

# Migration of mated (fertile) females carrying their pair genotype (i, j).
# probs must be column-normalised (adjacency_lists(A; normalize=true)).
function _migration_inter(Nf_pairs, neighbors, probs, w, mu_mig, M2, Nsites, ::Type{T}) where {T}
    Nf_new = zeros(T, M2, M2, Nsites)
    maxk   = maximum(length, neighbors; init=0)
    CI     = CartesianIndices((M2, M2))
    # Each (i, j) pair writes only to Nf_new[i, j, :] → safe to split over pairs
    _tforeach_chunk(M2 * M2) do r
        buf = Vector{T}(undef, maxk)
        @inbounds for idx in r
            i, j = Tuple(CI[idx])
            for site in 1:Nsites
                neigh = neighbors[site]
                k = length(neigh)
                k == 0 && continue
                cnt = round(T, mu_mig * Nf_pairs[i, j, site] * w[site])
                if cnt > 0
                    s = view(buf, 1:k)
                    rand!(Multinomial(cnt, probs[site]), s)
                    for q in 1:k
                        Nf_new[i, j, neigh[q]] += s[q]
                    end
                end
            end
        end
    end
    return Nf_new
end

function Migration_inter(; Nf_pairs, A=nothing, w, Nsites=size(Nf_pairs, 3), mu_mig=1,
                         M2=10, tp=Int64, neighbors=nothing, probs=nothing)
    if neighbors === nothing
        nb, pr = adjacency_lists(A; normalize=true)
        return _migration_inter(Nf_pairs, nb, pr, w, mu_mig, M2, Nsites, tp)
    end
    return _migration_inter(Nf_pairs, neighbors, probs, w, mu_mig, M2, Nsites, tp)
end


#==============================================================================
  5. NON-SPATIAL (PANMICTIC) MODELS
==============================================================================#

function Input_nosp(; M=4, Nmean=10^5, pm=0.01, tp=Int64)
    M2 = Int(M * (M + 1) / 2)
    Ntf_init = zeros(tp, M2)
    Ntf_init[1] = tp(round(Nmean / 2))
    Ntm_init = zeros(tp, M2)
    Ntm_init[1] = tp(round(Nmean / 2))
    Ntm_init[2] = ceil(tp, Ntm_init[1] * pm)
    Ntm_init[1] = ceil(tp, Ntm_init[1] * (1 - pm))
    return Ntf_init, Ntm_init
end

# r_sex for the non-spatial models (M = 4, 5, 6), with optional b2 override
function _r_sex_nosp(M, M2, mD_sex, b2)
    r_sex = copy(_make_r_sex(M, M2, mD_sex))
    if b2 > 0
        M == 4 && (r_sex[7, :] .= 0.5)
        M == 5 && (r_sex[9, :] .= 0.5)
    end
    return r_sex
end

# Tracks genotype counts every generation
function Model_nosp_T(; N=10^15, pm=0.01, wmf, e, mD_sex, beta=0, alpha=0, phi1=1, phi2=1,
                      maxgen=10, Model=2, M=4, tp=Int64, b2=0, wa=1)
    Ntf_init, Ntm_init = Input_nosp(; M=M, Nmean=N, pm=pm, tp=tp)
    M2 = Int(M * (M + 1) / 2)
    f = wmf[3][1, 1]
    alpha = alpha == 0 ? Alpha(phi1=phi1, phi2=phi2, f=f, N=N) : alpha
    r_sex = _r_sex_nosp(M, M2, mD_sex, b2)
    model = Model == 2 ? Model2 : Model1

    Ntf, Ntm = Array{Vector{tp}}(undef, maxgen), Array{Vector{tp}}(undef, maxgen)
    Ntf[1], Ntm[1] = copy(Ntf_init), copy(Ntm_init)
    NCountF = sum(Ntf_init)
    NCountM = sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        if NCountM > 1 && NCountF > 1
            # NOTE (unchanged): `beta` is overwritten every generation, so the
            # input beta only affects generation 1. Check this is intended.
            beta = NCountM / (NCountM + beta)
            m = model(; Nf=Ntf[gen], Nm=Ntm[gen], E=e, alpha=alpha, ff=wmf[3], wm=wmf[1],
                      wf=wmf[2], r_sex=r_sex, phi1=phi1, phi2=phi2, beta=beta, M2=M2, M=M,
                      tp=tp, wa=wa)
            Ntf[gen+1] = m[1]
            Ntm[gen+1] = m[2]
        else
            Ntf[gen+1], Ntm[gen+1] = zeros(tp, M2), zeros(tp, M2)
        end
        NCountM = sum(Ntm[gen+1])
        NCountF = sum(Ntf[gen+1])
    end
    return [Ntf, Ntm]
end

# Tracks only total females / males every generation
function Model_nosp_NT(; N=10^15, pm=0.01, wmf, e, mD_sex, beta=0, alpha=0, phi1=1, phi2=1,
                       maxgen=10, Model=2, M=4, tp=Int64, b2=0, wa=1)
    Ntf_init, Ntm_init = Input_nosp(; M=M, Nmean=N, pm=pm, tp=tp)
    M2 = Int(M * (M + 1) / 2)
    f = wmf[3][1, 1]
    alpha = alpha == 0 ? Alpha(phi1=phi1, phi2=phi2, f=f, N=N) : alpha
    r_sex = _r_sex_nosp(M, M2, mD_sex, b2)
    model = Model == 2 ? Model2 : Model1

    NCountF = Array{tp}(undef, maxgen)
    NCountM = Array{tp}(undef, maxgen)
    Ntf, Ntm = copy(Ntf_init), copy(Ntm_init)
    NCountF[1] = sum(Ntf_init)
    NCountM[1] = sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        if NCountM[gen] > 1 && NCountF[gen] > 1
            beta = NCountM[gen] / (NCountM[gen] + beta)   # NOTE: overwritten each gen (unchanged)
            m = model(; Nf=Ntf, Nm=Ntm, E=e, alpha=alpha, ff=wmf[3], wm=wmf[1], wf=wmf[2],
                      r_sex=r_sex, phi1=phi1, phi2=phi2, beta=beta, M2=M2, M=M, tp=tp, wa=wa)
            Ntf = m[1]
            Ntm = m[2]
        else
            Ntf, Ntm = zeros(tp, M2), zeros(tp, M2)
        end
        NCountM[gen+1] = sum(Ntm)
        NCountF[gen+1] = sum(Ntf)
    end
    return [NCountF, NCountM]
end


#==============================================================================
  6. SPATIAL MODELS
  Each public function keeps its original keyword signature, does the set-up,
  then calls a typed kernel. Inside a generation, sites are processed in
  parallel (they are independent until migration).
==============================================================================#

# Reproduction at every site (threaded over sites). Writes Ntf_next/Ntm_next
# and stores each site's current female total in sumf.
function _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model::F, E, alpha,
                        wmf, r_sex, phi1, phi2, wa, beta, M2, M, ::Type{T}) where {F,T}
    _tforeach_chunk(size(Ntf_cur, 2)) do r
        @inbounds for site in r
            Nf = view(Ntf_cur, :, site)
            Nm = view(Ntm_cur, :, site)
            sm, sf = sum(Nm), sum(Nf)
            sumf[site] = sf
            if sm > 0 && sf > 0
                beta0 = sm / (sm + beta)
                m = model(; Nf=Nf, Nm=Nm, E=E, alpha=alpha[site], ff=wmf[3], wm=wmf[1],
                          wf=wmf[2], r_sex=r_sex, phi1=phi1, phi2=phi2, wa=wa, beta=beta0,
                          M2=M2, M=M, tp=T)
                Ntf_next[:, site] .= m[1]
                Ntm_next[:, site] .= m[2]
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end
        end
    end
    return nothing
end

# Allocate the output of Model_sp / Model_sp_inter
function _sp_output(stats_only, track, maxgen, Nsites, M2, ::Type{T}) where {T}
    stats_only && return zeros(Float64, maxgen, Nsites)
    return track ? Array{T}(undef, 2, maxgen, Nsites, M2) : Array{T}(undef, 2, maxgen, Nsites)
end

# Record totals (track=false) or full genotype vectors (track=true)
function _sp_record!(Nt_mat, gen, Ntf_next, Ntm_next, track)
    @inbounds for site in axes(Ntf_next, 2)
        if !track
            Nt_mat[1, gen, site] = sum(view(Ntf_next, :, site))
            Nt_mat[2, gen, site] = sum(view(Ntm_next, :, site))
        else
            @views Nt_mat[1, gen, site, :] .= Ntf_next[:, site]
            @views Nt_mat[2, gen, site, :] .= Ntm_next[:, site]
        end
    end
    return nothing
end

# ---------- 6a. Model_sp: adult migration after reproduction ----------------

function _kernel_sp(model::F, Ntf_init, Ntm_init, neighbors, probs, w_scaled, wmf, E, r_sex,
                    alpha, wa, phi1, phi2, maxgen, M, M2, beta, ::Type{T}, track,
                    stats_only) where {F,T}
    Nsites = size(Ntf_init, 2)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    sumf = zeros(eltype(Ntf_cur), Nsites)
    nsf  = vec(sum(Ntf_init; dims=1))            # initial females per site
    out  = _sp_output(stats_only, track, maxgen, Nsites, M2, T)

    for gen in 1:maxgen
        _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model, E, alpha, wmf,
                       r_sex, phi1, phi2, wa, beta, M2, M, T)
        if stats_only
            # NOTE (unchanged): uses females BEFORE this generation's reproduction,
            # whereas Model_sp_inter records females AFTER it.
            @inbounds for site in 1:Nsites
                out[gen, site] = sumf[site] / nsf[site]
            end
        else
            _sp_record!(out, gen, Ntf_next, Ntm_next, track)
        end
        if gen < maxgen
            Nf_mig, Nm_mig = _migration(Ntf_next, Ntm_next, neighbors, probs, w_scaled, T)
            Ntf_cur .= Nf_mig
            Ntm_cur .= Nm_mig
        end
    end
    return out
end

function Model_sp(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
                  alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1,
                  mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
                  track=false, stats_only=true)
    M2 = div(M * (M + 1), 2)
    f  = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)
    r_sex   = _make_r_sex(M, M2, mD_sex)
    model   = Model == 2 ? Model2 : Model1
    neighbors, probs = adjacency_lists(Adj[1])
    return _kernel_sp(model, Ntf_init, Ntm_init, neighbors, probs, mu_mig .* Adj[2], wmf, E,
                      r_sex, alpha_v, wa, phi1, phi2, maxgen, M, M2, beta, tp, track, stats_only)
end

# ---------- 6b. Model_sp_inter: mated females migrate between mating and egg-laying

# Phase 1: form fertile pairs at each site (threaded over sites)
function _mating_phase1!(N_pairs_Af, N_pairs_A, Nf, Nm, beta, ff, wa, M2, ::Type{T}) where {T}
    Nm_w   = Nm .* wa
    Nm_sum = sum(Nm_w)
    pAm    = Nm_sum > 0 ? Nm_w ./ Nm_sum : fill(1 / M2, M2)   # avoid division by zero
    p_beta = Nm_sum / (beta + Nm_sum)
    fill!(N_pairs_A, zero(T))
    @inbounds for i in 1:M2
        n = round(T, Nf[i] * p_beta)
        n > 0 && rand!(Multinomial(n, pAm), view(N_pairs_A, i, :))
    end
    @inbounds for j in 1:M2, i in 1:M2
        a = N_pairs_A[i, j]
        N_pairs_Af[i, j] = a > 0 ? rand(Poisson(a * ff[i, j])) : zero(T)
    end
    return N_pairs_Af
end

function Mating_phase1(; Nf, Nm, beta=0, ff, M2=10, tp=Int64, wa=1)
    out = zeros(tp, M2, M2)
    return _mating_phase1!(out, zeros(tp, M2, M2), Nf, Nm, beta, ff, wa, M2, tp)
end

# Phase 2: offspring from the (migrated) fertile pairs at one site
function _mating_phase2!(N_exp_f, N_exp_m, N_pairs_Af, E, M2, r_sex, buf)
    fill!(N_exp_f, 0)
    fill!(N_exp_m, 0)
    @inbounds for j in 1:M2, i in 1:M2
        n = N_pairs_Af[i, j]
        n > 0 && _offspring_pair!(N_exp_f, N_exp_m, n, E[i][j], 1 - r_sex[j, i], buf)
    end
    return N_exp_f, N_exp_m
end

function Mating_phase2(; N_pairs_Af, E, M2, r_sex, tp=Int64)
    return _mating_phase2!(zeros(tp, M2), zeros(tp, M2), N_pairs_Af, E, M2, r_sex, zeros(tp, M2))
end

function _inter_phase1!(pairs1, Ntf_cur, Ntm_cur, beta, ff, wa, M2, ::Type{T}) where {T}
    _tforeach_chunk(size(Ntf_cur, 2)) do r
        N_pairs_A = zeros(T, M2, M2)                 # per-task scratch
        @inbounds for site in r
            Nf  = view(Ntf_cur, :, site)
            Nm  = view(Ntm_cur, :, site)
            out = view(pairs1, :, :, site)
            sm, sf = sum(Nm), sum(Nf)
            if sm > 0 && sf > 0
                beta0 = sm / (sm + beta)
                _mating_phase1!(out, N_pairs_A, Nf, Nm, beta0, ff, wa, M2, T)
            else
                fill!(out, zero(T))
            end
        end
    end
    return nothing
end

function _inter_phase2!(Ntf_next, Ntm_next, pairs2, Model, E, r_sex, alpha, wmf, phi1, phi2,
                        M2, ::Type{T}) where {T}
    _tforeach_chunk(size(Ntf_next, 2)) do r
        buf = zeros(T, M2)
        Nf  = zeros(T, M2)
        Nm  = zeros(T, M2)
        @inbounds for site in r
            P = view(pairs2, :, :, site)
            if sum(P) > 0
                _mating_phase2!(Nf, Nm, P, E, M2, r_sex, buf)
                if Model == 2        # density dependence → fitness
                    Nh = sum(Nf) + sum(Nm)
                    Nf1, Nm1 = _ddm_change(Nf, Nm, Nh, alpha[site], phi1, phi2, M2, T)
                    Nf2, Nm2 = _fitness_change(Nf1, Nm1, wmf[2], wmf[1], M2, T)
                else                 # fitness → density dependence
                    Nf1, Nm1 = _fitness_change(Nf, Nm, wmf[2], wmf[1], M2, T)
                    Nh = sum(Nf1) + sum(Nm1)
                    # RESULT CHANGE: original applied this to (Nf, Nm), discarding fitness
                    Nf2, Nm2 = _ddm_change(Nf1, Nm1, Nh, alpha[site], phi1, phi2, M2, T)
                end
                Ntf_next[:, site] .= Nf2
                Ntm_next[:, site] .= Nm2
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end
        end
    end
    return nothing
end

function _kernel_sp_inter(Model, Ntf_init, Ntm_init, neighbors, probs_n, w, mu_mig, wmf, E,
                          r_sex, alpha, wa, phi1, phi2, maxgen, M2, beta, ::Type{T}, track,
                          stats_only) where {T}
    Nsites = size(Ntf_init, 2)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    pairs1 = zeros(T, M2, M2, Nsites)
    nsf = vec(sum(Ntf_init; dims=1))
    out = _sp_output(stats_only, track, maxgen, Nsites, M2, T)

    for gen in 1:maxgen
        # RESULT CHANGE (only if wa != 1): wa is now used for Model=2 as well
        _inter_phase1!(pairs1, Ntf_cur, Ntm_cur, beta, wmf[3], wa, M2, T)
        pairs2 = _migration_inter(pairs1, neighbors, probs_n, w, mu_mig, M2, Nsites, T)
        _inter_phase2!(Ntf_next, Ntm_next, pairs2, Model, E, r_sex, alpha, wmf, phi1, phi2, M2, T)

        if stats_only
            @inbounds for site in 1:Nsites
                out[gen, site] = sum(view(Ntf_next, :, site)) / nsf[site]
            end
        else
            _sp_record!(out, gen, Ntf_next, Ntm_next, track)
        end
        if gen < maxgen
            copy!(Ntf_cur, Ntf_next)
            copy!(Ntm_cur, Ntm_next)
        end
    end
    return out
end

function Model_sp_inter(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E,
                        mD_sex=0.5, alpha=0, phi1=1, phi2=1, maxgen=100, bound=1, Model=2,
                        mu_mig=1, maxdist=1, M=4, M2=0, beta=0, fs=[], wa=1,
                        tp=Int64, track=false, stats_only=true)
    if M2 < 1
        M2 = div(M * (M + 1), 2)
    end
    f = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)
    r_sex   = _make_r_sex(M, M2, mD_sex)
    neighbors, probs_n = adjacency_lists(Adj[1]; normalize=true)
    return _kernel_sp_inter(Model, Ntf_init, Ntm_init, neighbors, probs_n, Adj[2], mu_mig, wmf,
                            E, r_sex, alpha_v, wa, phi1, phi2, maxgen, M2, beta, tp, track,
                            stats_only)
end

# ---------- 6c. Model_spO: frequency of chosen allele(s) per site -----------

# Weights on females (k1) and males (k2) for the allele statistics.
# NOTE (unchanged): sex=:b gives males weight 2 (k2 = 2). Check this is intended.
_sex_weights(sex) = sex == :b ? (1, 2) : sex == :f ? (1, 0) : (0, 1)

function _kernel_spO(model::F, Ntf_init, Ntm_init, neighbors, probs, w_scaled, wmf, E, r_sex,
                     alpha, wa, phi1, phi2, maxgen, M, M2, beta, ::Type{T}, R1Ind, k1, k2,
                     freq) where {F,T}
    Nsites = size(Ntf_init, 2)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    sumf = zeros(eltype(Ntf_cur), Nsites)
    nsf  = vec(sum(Ntf_init; dims=1))
    Stat1 = zeros(Float64, maxgen, Nsites)

    for gen in 1:maxgen
        _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model, E, alpha, wmf,
                       r_sex, phi1, phi2, wa, beta, M2, M, T)
        @inbounds for site in 1:Nsites
            num = sum(view(Ntf_next, R1Ind, site)) * k1 + sum(view(Ntm_next, R1Ind, site)) * k2
            if !freq     # relative to initial female count
                Stat1[gen, site] = num / (nsf[site] * (k1 + k2))
            else         # frequency in the current population
                Stat1[gen, site] = num / (sum(view(Ntf_next, :, site)) * k1 +
                                          sum(view(Ntm_next, :, site)) * k2)
            end
        end
        if gen < maxgen
            Nf_mig, Nm_mig = _migration(Ntf_next, Ntm_next, neighbors, probs, w_scaled, T)
            Ntf_cur .= Nf_mig
            Ntm_cur .= Nm_mig
        end
    end
    return Stat1
end

function Model_spO(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
                   alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1, R1Ind, sex=:b,
                   freq=false, mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
                   track=false, stats_only=true)
    M2 = div(M * (M + 1), 2)
    f  = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)
    model   = Model == 2 ? Model2 : Model1
    k1, k2  = _sex_weights(sex)
    r_sex   = _make_r_sex(M, M2, mD_sex)
    neighbors, probs = adjacency_lists(Adj[1])     # FIX: old call used removed keywords
    return _kernel_spO(model, Ntf_init, Ntm_init, neighbors, probs, mu_mig .* Adj[2], wmf, E,
                       r_sex, alpha_v, wa, phi1, phi2, maxgen, M, M2, beta, tp, R1Ind, k1, k2,
                       freq)
end

# ---------- 6d. Model_spO_cut: stop when the allele statistic crosses threshold

function _kernel_spO_cut(model::F, Ntf_init, Ntm_init, neighbors, probs, w_scaled, wmf, E,
                         r_sex, alpha, wa, phi1, phi2, maxgen, M, M2, beta, ::Type{T}, R1Ind,
                         k1, k2, to_first, threshold, p) where {F,T}
    Nsites = size(Ntf_init, 2)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    sumf  = zeros(eltype(Ntf_cur), Nsites)
    nsf   = vec(sum(Ntf_init; dims=1))
    denom = nsf .* (k1 + k2)

    found_gen, found_site = nothing, nothing
    found_gen2  = zeros(Nsites)     # last generation each site was at or below p
    found_site2 = zeros(Nsites)     # that site's initial female count

    for gen in 1:maxgen
        # Reproduce all sites in parallel, then scan in site order so the
        # early-exit result is exactly what the serial version would return.
        _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model, E, alpha, wmf,
                       r_sex, phi1, phi2, wa, beta, M2, M, T)
        @inbounds for site in 1:Nsites
            if sum(view(Ntf_next, :, site)) / nsf[site] <= p
                found_gen2[site]  = gen
                found_site2[site] = nsf[site]
            end
            stat_val = (sum(view(Ntf_next, R1Ind, site)) * k1 +
                        sum(view(Ntm_next, R1Ind, site)) * k2) / denom[site]
            if to_first && stat_val >= threshold
                return gen, site, found_gen2, found_site2
            end
        end
        if gen < maxgen
            Nf_mig, Nm_mig = _migration(Ntf_next, Ntm_next, neighbors, probs, w_scaled, T)
            Ntf_cur .= Nf_mig
            Ntm_cur .= Nm_mig
        end
    end
    return found_gen, found_site, found_gen2, found_site2
end

function Model_spO_cut(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
                       alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1, R1Ind, sex=:b,
                       mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
                       track=false, stats_only=true, to_first=true, threshold=1, p=0.33)
    M2 = div(M * (M + 1), 2)
    f  = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)
    model   = Model == 2 ? Model2 : Model1
    k1, k2  = _sex_weights(sex)
    r_sex   = _make_r_sex(M, M2, mD_sex)
    neighbors, probs = adjacency_lists(Adj[1])     # FIX: old call used removed keywords
    return _kernel_spO_cut(model, Ntf_init, Ntm_init, neighbors, probs, mu_mig .* Adj[2], wmf,
                           E, r_sex, alpha_v, wa, phi1, phi2, maxgen, M, M2, beta, tp, R1Ind,
                           k1, k2, to_first, threshold, p)
end

# ---------- 6e. Model_spO_cut_flagged ---------------------------------------
# Returns (first generation the site was continuously flagged, site) for the
# first site, in site order, whose allele statistic reaches `threshold`.
# Returns (nothing, nothing) if a site in SInds reaches `threshold` for the
# D1Ind genotypes first, or if maxgen passes without a hit.

function _kernel_spO_cut_flagged(model::F, Ntf_init, Ntm_init, neighbors, probs, w_scaled, wmf,
                                 E, r_sex, alpha, wa, phi1, phi2, maxgen, M, M2, beta,
                                 ::Type{T}, R1Ind, D1Ind, isS, k1, k2, to_first,
                                 threshold) where {F,T}
    Nsites = size(Ntf_init, 2)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    sumf  = zeros(eltype(Ntf_cur), Nsites)
    nsf   = vec(sum(Ntf_init; dims=1))
    denom = nsf .* (k1 + k2)

    flagged  = falses(Nsites)       # site currently has the allele (stat > 0)
    flag_gen = zeros(Int, Nsites)   # generation the current flag started

    for gen in 1:maxgen
        # Reproduce all sites in parallel, then scan in site order so flags and
        # early exits are exactly what the original serial loop produced.
        _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model, E, alpha, wmf,
                       r_sex, phi1, phi2, wa, beta, M2, M, T)
        @inbounds for site in 1:Nsites
            stat_val = (sum(view(Ntf_next, R1Ind, site)) * k1 +
                        sum(view(Ntm_next, R1Ind, site)) * k2) / denom[site]

            # Stop (no result) if a D1Ind genotype reaches threshold at an SInds site
            if isS[site]
                d1_val = (sum(view(Ntf_next, D1Ind, site)) * k1 +
                          sum(view(Ntm_next, D1Ind, site)) * k2) / denom[site]
                d1_val >= threshold && return nothing, nothing
            end

            if stat_val > 0
                if !flagged[site]
                    flagged[site]  = true
                    flag_gen[site] = gen
                end
                if to_first && stat_val >= threshold
                    return flag_gen[site], site
                end
            else
                flagged[site]  = false
                flag_gen[site] = 0
            end
        end
        if gen < maxgen
            Nf_mig, Nm_mig = _migration(Ntf_next, Ntm_next, neighbors, probs, w_scaled, T)
            Ntf_cur .= Nf_mig
            Ntm_cur .= Nm_mig
        end
    end
    return nothing, nothing
end

function Model_spO_cut_flagged(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
                               alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1, R1Ind, D1Ind,
                               SInds, sex=:b, mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
                               track=false, stats_only=true, to_first=true, threshold=1,
                               break_gen=70)            # break_gen: accepted but unused (as before)
    M2 = M * (M + 1) ÷ 2
    f  = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)   # FIX: per-site sum
    model   = Model == 2 ? Model2 : Model1
    k1, k2  = _sex_weights(sex)
    r_sex   = _make_r_sex(M, M2, mD_sex)
    isS     = falses(size(Ntf_init, 2))
    isS[SInds] .= true
    neighbors, probs = adjacency_lists(Adj[1])      # works for the dense Test_input matrix
    return _kernel_spO_cut_flagged(model, Ntf_init, Ntm_init, neighbors, probs, mu_mig .* Adj[2],
                                   wmf, E, r_sex, alpha_v, wa, phi1, phi2, maxgen, M, M2, beta, tp,
                                   R1Ind, D1Ind, isS, k1, k2, to_first, threshold)
end


#==============================================================================
  7. MULTIPLEXING MODELS (non-spatial)
  Replicates are independent, so they now run in parallel on threads.
==============================================================================#

# Multiplexing 1 (M1), panmictic
function Model_nospM(; N=10^15, pm=0.01, wmf, e, mD_sex, beta=0, alpha=0, phi1=1, phi2=1,
                     maxgen=10, Model=2, M=4, tp=Int64, track=false, wa=1)
    Ntf_init, Ntm_init = InputMulti(N; M=M, pm=pm, tp=tp)
    M  = M - 1
    M1 = Int(M * (M + 1) ÷ 2) + 1
    M2 = Int(M1 * (M1 + 1) ÷ 2)
    f  = wmf[3][1]
    alpha = alpha == 0 ? Alpha(; N=N, f=f, phi1=phi1, phi2=phi2) : alpha

    r_sex = 0.5 .* ones(M2, M2)
    for i in (M2 - M1 + 1):M2          # genotypes carrying the drive
        r_sex[i, :] .= mD_sex
    end
    model = Model == 2 ? Model2 : Model1   # was always Model2 (callers pass Model=2)

    NCountF = Array{tp}(undef, maxgen)
    NCountM = Array{tp}(undef, maxgen)
    Ntf, Ntm = copy(Ntf_init), copy(Ntm_init)
    NCountF[1] = sum(Ntf_init)
    NCountM[1] = sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        if NCountM[gen] > 1 && NCountF[gen] > 1
            beta = NCountM[gen] / (NCountM[gen] + beta)   # NOTE: overwritten each gen (unchanged)
            m = model(; Nf=Ntf, Nm=Ntm, E=e, alpha=alpha, ff=wmf[3], wm=wmf[1], wf=wmf[2],
                      r_sex=r_sex, beta=beta, M2=M2, M=M1, tp=tp, phi1=phi1, phi2=phi2, wa=wa)
            Ntf = m[1]
            Ntm = m[2]
        else
            Ntf, Ntm = zeros(tp, M2), zeros(tp, M2)
        end
        # NOTE (unchanged): with track=true the counts are never filled in
        if !track
            NCountM[gen+1] = sum(Ntm)
            NCountF[gen+1] = sum(Ntf)
        end
    end
    return [NCountF, NCountM]
end

# Multiplexing 1: replicate runner → saves 2 .jld2 files (models, outcomes)
function NospM_run_NT(; SexDistortion=0.5, B=0.001, C=0.95, J=0.035, sigma=0.01, s=1, h=0,
                      hDR=0, hm=0, hN=0, sN=1, F=12, phi1=1, phi2=1, wa=1, Npop=10^8, rep=500,
                      maxgen=500, Release=0.1, M=3, filename="Multi", homing=:b, beta=0,
                      showprogress=false, dir="", tp=Int64)
    tag = string("_Npop_", log10.(Npop), "_B_", B, "_J_", J, "_C_", C, "h_", h, "_m_", SexDistortion, ".jld2")
    fn  = string(dir, filename, tag)
    fn2 = string(dir, "Outcome", filename, tag)

    e   = E_Make_Multi(; c_m=C, c_f=C, j_m=J, j_f=J, b_m=B, b_f=B, M=M, homing=homing)
    wmf = FitnessMulti2(; sigma=sigma, s=s, h=h, hN=hN, sN=sN, F=F)

    R1_temp    = Vector{Any}(undef, rep)
    R1_outcome = Vector{Any}(undef, rep)
    prog = Progress(rep; enabled=showprogress)

    _tforeach_chunk(rep) do r
        for rp in r
            mod = Model_nospM(; N=Npop, wmf=wmf, e=e, mD_sex=SexDistortion, M=M, maxgen=maxgen,
                              Model=2, wa=wa, phi1=phi1, phi2=phi2, pm=Release, beta=beta, tp=tp)
            R1_temp[rp]    = mod
            R1_outcome[rp] = last.(mod)   # FIX: previously empty when showprogress=true
            next!(prog)
        end
    end
    jldsave(fn, true; models1=R1_temp)
    jldsave(fn2, true; res=R1_outcome)
end

# Multiplexing 2 haplotype names: Names2 (13 two-locus haplotypes) and the 91
# genotype labels "A||B" in the order used by the gamete and fitness matrices.
function _multi2_names()
    Names2  = ["WW", "WN", "WR", "NW", "NN", "NR", "RW", "RN", "RR", "DO", "OD", "DD", "OO"]
    n = length(Names2)
    HHnames = reshape([string(n1, "||", n2) for n2 in Names2 for n1 in Names2], n, n)
    MM_Ind  = collect(zip(reduce(vcat, [1:i for i in 1:n]), reduce(vcat, [repeat([i], i) for i in 1:n])))
    Haplotypes = [HHnames[MM_Ind[j][1], MM_Ind[j][2]] for j in eachindex(MM_Ind)]
    return Names2, Haplotypes, MM_Ind
end

# Multiplexing 2: replicate runner → saves 2 .jld2 files (models, outcomes)
function NospMulti2_run_NT(; SexDistortion=0.5,
        k=0.5, c1=0.995, c11=100, c2=0.995, j1=0.035, j2=0.035, b1=0.001, phi1=1, phi2=1, wa=1,
        delta1=[1, 0, 0, 0], delta=[1, 0, 0, 0], deltaA=0.0009, deltaB=0.0032, l=0.5, useQFS=true,
        QFS=2, showF=false, showM=false, homing=:b, precision=4,
        sigma=0.01, s=1, h=0.3, hDR=0.02, hN=0.02, sN=1, hm=0, F=12, showValues=false,
        showSummary=false, showFormula=false,
        Npop=10^8, rep=500, maxgen=500, Release=0.1, M=13, filename="Multi2_", beta=0,
        showprogress=false, dir="", tp=Int64)

    Names2, Haplotypes, _ = _multi2_names()
    Ntf_init, Ntm_init = InputMulti2(Npop; M=M, pm=Release, namesH=Haplotypes, tp=tp)

    tag = string("_Npop_", log10.(Npop), "_B_", b1, "_J_", j1, "_C1_", c1, "k_", k, "h_", h, "_m_", SexDistortion, ".jld2")
    fn  = string(dir, filename, tag)
    fn2 = string(dir, "Outcome", filename, tag)

    if useQFS
        QFS == 2 && println("Using QFS2 data")
        QFS == 5 && println("Using QFS5 data")
    end

    e = Gametes_all(; k=k, l=l, c1=c1, c11=c11, c2=c2, j1=j1, j2=j2, b1=b1, delta1=delta1,
                    delta=delta, deltaA=deltaA, deltaB=deltaB, useQFS=useQFS, QFS=QFS,
                    showF=showF, showM=showM, homing=homing, precision=precision)
    wmf = FitnessMulti22(; sigma=sigma, s=s, h=h, hDR=hDR, hN=hN, sN=sN, hm=hm, F=F,
                         Hnames=Names2, showFormula=showFormula, showValues=showValues,
                         showSummary=showSummary)

    R1_temp    = Vector{Any}(undef, rep)
    R1_outcome = Vector{Any}(undef, rep)
    prog = Progress(rep; enabled=showprogress)

    _tforeach_chunk(rep) do r
        for rp in r
            mod = Multi2_nosp_NT(; Ntf_init=Ntf_init, Ntm_init=Ntm_init, wmf=wmf, e=e,
                                 mD_sex=SexDistortion, M=M, maxgen=maxgen, Model=2, wa=wa,
                                 phi1=phi1, phi2=phi2, beta=beta, tp=tp, namesH=Haplotypes)
            R1_temp[rp]    = mod
            R1_outcome[rp] = last.(mod)   # FIX: previously empty when showprogress=true
            next!(prog)
        end
    end
    jldsave(fn, true; models1=R1_temp)
    jldsave(fn2, true; res=R1_outcome)
end

function Multi2_nosp_NT(; Ntf_init, Ntm_init, wmf, e, mD_sex, beta=0, alpha=0, maxgen=500,
                        phi1=1, phi2=1, Model=2, wa=1, M=13, namesH=_multi2_names()[2], tp=Int64)
    M2 = Int(M * (M + 1) / 2)   # 91 for M = 13
    f  = wmf[3][1]
    alpha = alpha == 0 ? Alpha(N=sum(Ntf_init .+ Ntm_init), f=f, phi1=phi1, phi2=phi2) : alpha

    r_sex = 0.5 .* ones(M2, M2)
    D_Ind = findall(occursin.("D", namesH))    # FIX: used global `Haplotypes`
    for i in D_Ind
        r_sex[i, :] .= mD_sex
    end
    model = Model == 2 ? Model2 : Model1       # was always Model2 (callers pass Model=2)

    NCountF = Array{tp}(undef, maxgen)
    NCountM = Array{tp}(undef, maxgen)
    Ntf, Ntm = copy(Ntf_init), copy(Ntm_init)
    NCountF[1] = sum(Ntf_init)
    NCountM[1] = sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        if NCountM[gen] > 1 && NCountF[gen] > 1
            beta = NCountM[gen] / (NCountM[gen] + beta)   # NOTE: overwritten each gen (unchanged)
            m = model(; Nf=Ntf, Nm=Ntm, E=e, alpha=alpha, ff=wmf[3], wm=wmf[1], wf=wmf[2],
                      r_sex=r_sex, phi1=phi1, phi2=phi2, beta=beta, wa=wa, M2=M2, M=M, tp=tp)
            Ntf = m[1]
            Ntm = m[2]
        else
            Ntf, Ntm = zeros(tp, M2), zeros(tp, M2)
        end
        NCountM[gen+1] = sum(Ntm)
        NCountF[gen+1] = sum(Ntf)
    end
    return [NCountF, NCountM]
end


#==============================================================================
  8. DISTRIBUTED DRIVERS (one replicate per pmap task, saved to its own file)
  CachingPool sends the (large) closure to each worker once, not per replicate.
  run_replicate* live here, so `@everywhere include` defines them on workers.
==============================================================================#

function run_replicate(i; wmf, SexDistortion, e, phi1, phi2, Model, Nmv, Nfv, maxgen, Adj,
                       Release, alpha, wa, Release_sites, NumRelease, M, homing, beta, track,
                       stats_only, fn_prefix)
    mod = Model_sp_inter(; Ntf_init=Nfv, Ntm_init=Nmv, Adj=Adj, wmf=wmf, E=e, Model=Model,
                         alpha=alpha, maxgen=maxgen, phi1=phi1, phi2=phi2, wa=wa, mu_mig=1,
                         Release_sites=Release_sites, maxdist=1, M=M, mD_sex=SexDistortion,
                         beta=beta, track=track, stats_only=stats_only)
    fn = string(fn_prefix, "_rep_", i, ".jld2")
    @save fn mod
    return fn
end

function run_replicateM1(i; wmf, Adj, r_sex, Nfv, Nmv, maxgen, E, alpha, Release, Release_sites,
                         phi1, phi2, wa, M2, homing, beta, track, stats_only, fn_prefix)
    mod = Model_sp_inter(; Ntf_init=Nfv, Ntm_init=Nmv, maxgen=maxgen, mu_mig=1, wmf=wmf, E=E,
                         Adj=Adj, alpha=alpha, wa=wa, Release_sites=Release_sites, maxdist=1,
                         M2=M2, mD_sex=r_sex, phi1=phi1, phi2=phi2, beta=beta, track=track,
                         stats_only=stats_only)
    fn = string(fn_prefix, "_rep_", i, ".jld2")
    @save fn mod
    return fn
end

function run_replicateO(i; wmf, SexDistortion, e, phi1, phi2, freq, Nmv, Nfv, maxgen, Adj,
                        Release, alpha, wa, sex, Release_sites, NumRelease, M, homing, R1Ind,
                        beta, track, stats_only, fn_prefix)
    mod = Model_spO(; freq=freq, Ntf_init=Nfv, Ntm_init=Nmv, Adj=Adj, wmf=wmf, E=e, alpha=alpha,
                    maxgen=maxgen, phi1=phi1, phi2=phi2, wa=wa, R1Ind=R1Ind, mu_mig=1,
                    Release_sites=Release_sites, maxdist=1, M=M, mD_sex=SexDistortion, sex=sex,
                    beta=beta, track=track, stats_only=stats_only)
    fn = string(fn_prefix, "_R1source_rep_", i, ".jld2")
    @save fn mod
    return fn
end

function run_replicateO_cut(i; wmf, SexDistortion, e, phi1, phi2, threshold, p, Nmv, Nfv,
                            maxgen, Adj, Release, alpha, wa, sex, Model, Release_sites,
                            NumRelease, M, homing, R1Ind, beta, track, stats_only, fn_prefix)
    mod = Model_spO_cut(; threshold=threshold, p=p, Ntf_init=Nfv, Ntm_init=Nmv, Adj=Adj, wmf=wmf,
                        E=e, alpha=alpha, maxgen=maxgen, Model=Model, phi1=phi1, phi2=phi2,
                        wa=wa, R1Ind=R1Ind, mu_mig=1, Release_sites=Release_sites, maxdist=1,
                        M=M, mD_sex=SexDistortion, sex=sex, beta=beta, track=track,
                        stats_only=stats_only)
    fn = string(fn_prefix, "_R1sourceCut_rep_", i, ".jld2")
    @save fn mod
    return fn
end

# Model_spO_cut_flagged is defined in section 6e
function run_replicateO_cut_flagged(i; wmf, SexDistortion, e, phi1, phi2, threshold, Nmv, Nfv,
                                    maxgen, Adj, Release, alpha, wa, sex, Model, Release_sites,
                                    NumRelease, M, homing, R1Ind, D1Ind, SInds, beta, track,
                                    stats_only, fn_prefix)
    mod = Model_spO_cut_flagged(; threshold=threshold, Ntf_init=Nfv, Ntm_init=Nmv, Adj=Adj,
                                wmf=wmf, E=e, alpha=alpha, maxgen=maxgen, Model=Model, phi1=phi1,
                                phi2=phi2, wa=wa, R1Ind=R1Ind, D1Ind=D1Ind, SInds=SInds, mu_mig=1,
                                Release_sites=Release_sites, maxdist=1, M=M, mD_sex=SexDistortion,
                                sex=sex, beta=beta, track=track, stats_only=stats_only)
    fn = string(fn_prefix, "_R1sourceCutFlag_rep_", i, ".jld2")
    @save fn mod
    return fn
end

# Genotype names used by the allele-statistic drivers
_names_genotypes(M) = M == 4 ?
    ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"] :
    ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]

function Sp_runDistr(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                     Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01, Model=2,
                     Release=0.1, Release_sites=[], NumRelease=10^3, alpha=0, phi1=1, phi2=1, wa=1,
                     M=4, filename="Sp", homing=:b, beta=100, track=false, i0=1, stats_only=true,
                     initN=false, Nfv0=[], Nmv0=[], Nfv=[], Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    if isempty(Release_sites)
        Release_sites = [dia]
    end
    f = wmf[3][1, 1]
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B, c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in   = 1 .- mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    tr  = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)

    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicate(i; wmf=wmf, Nmv=Nmv, Nfv=Nfv, Model=Model, SexDistortion=SexDistortion, e=e,
                      Adj=Adj, maxgen=maxgen, alpha=alpha, wa=wa, phi1=phi1, phi2=phi2,
                      Release=Release, Release_sites=Release_sites, NumRelease=NumRelease, M=M,
                      homing=homing, beta=beta, track=track, fn_prefix=fn_prefix,
                      stats_only=stats_only)
    end
end

function Sp_run_M1_Distr(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05,
                         Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01, alpha=0, phi1=1, phi2=1, wa=1,
                         Release=0.1, Release_sites=[], NumRelease=10^3,
                         M=4, filename="SpM1", homing=:b, beta=0, track=false, stats_only=true, i0=1,
                         initN=false, Nfv0=[], Nmv0=[], Nfv=[], Nmv=[])
    f  = wmf[3][1, 1]
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_f_", f, "_mig_", mig)

    M  = M - 1
    M1 = Int(M * (M + 1) ÷ 2) + 1
    M2 = Int(M1 * (M1 + 1) ÷ 2)
    if isempty(Release_sites)
        Release_sites = [dia]
    end
    p_in   = 1 .- mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    e   = E_Make_Multi(; c_m=C, j_m=J, b_m=B, c_f=C, j_f=J, b_f=B, M=M, homing=homing)

    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, ReleaseGT=M2, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, ReleaseGT=M2, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    r_sex = fill(0.5, M2, M2)
    for i in (M2 - M1 + 1):M2
        r_sex[i, :] .= SexDistortion
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicateM1(i; wmf=wmf, r_sex=r_sex, E=e, Adj=Adj, Nfv=Nfv, Nmv=Nmv, maxgen=maxgen,
                        alpha=alpha, phi1=phi1, phi2=phi2, wa=wa, Release=Release,
                        Release_sites=Release_sites, M2=M2, homing=homing, beta=beta, track=track,
                        fn_prefix=fn_prefix, stats_only=stats_only)
    end
end

# Presence of allele(s) relative to the initial population size
function Sp_runDistrO(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0, freq=false,
                      Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01, allele="R",
                      Release=0.1, Release_sites=[], NumRelease=10^3, alpha=0, phi1=1, phi2=1, wa=1, sex=:b,
                      M=4, filename="Sp", homing=:b, beta=100, track=false, i0=1, stats_only=true,
                      initN=false, Nfv0=[], Nmv0=[], Nfv=[], Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    namesG = _names_genotypes(M)
    R1Ind  = sum([occursin.(allele[i], namesG) for i in eachindex(allele)]) .> 0

    if isempty(Release_sites)
        Release_sites = [dia]
    end
    f = wmf[3][1, 1]
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B, c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in   = 1 - mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    tr  = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)

    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicateO(i; freq=freq, wmf=wmf, Nmv=Nmv, Nfv=Nfv, SexDistortion=SexDistortion, e=e,
                       Adj=Adj, maxgen=maxgen, alpha=alpha, wa=wa, phi1=phi1, phi2=phi2,
                       Release=Release, Release_sites=Release_sites, NumRelease=NumRelease, M=M,
                       sex=sex, homing=homing, R1Ind=R1Ind, beta=beta, track=track,
                       fn_prefix=fn_prefix, stats_only=stats_only)
    end
end

function Sp_runDistrO_cut(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                          Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01, allele="R", threshold=1, p=0.33, Model=2,
                          Release=0.1, Release_sites=[], NumRelease=10^3, alpha=0, phi1=1, phi2=1, wa=1, sex=:b,
                          M=4, filename="Sp", homing=:b, beta=100, track=false, i0=1, stats_only=true,
                          initN=false, Nfv0=[], Nmv0=[], Nfv=[], Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    namesG = _names_genotypes(M)
    R1Ind  = sum([occursin.(allele[i], namesG) for i in eachindex(allele)]) .> 0

    if isempty(Release_sites)
        Release_sites = [dia]
    end
    f = wmf[3][1, 1]
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B, c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in   = 1 - mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    tr  = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)

    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicateO_cut(i; threshold=threshold, p=p, wmf=wmf, Nmv=Nmv, Nfv=Nfv,
                           SexDistortion=SexDistortion, e=e, Adj=Adj, maxgen=maxgen, alpha=alpha,
                           wa=wa, phi1=phi1, phi2=phi2, Release=Release, Release_sites=Release_sites,
                           NumRelease=NumRelease, M=M, Model=Model, sex=sex, homing=homing,
                           R1Ind=R1Ind, beta=beta, track=track, fn_prefix=fn_prefix,
                           stats_only=stats_only)
    end
end

# Stops early if the D allele(s) reach `threshold` in the SInds band.
# Hex_band is in section 12.
function Sp_runDistrO_cut_flagged(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                                  Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01, allele="R", d_allele="D",
                                  threshold=1, Model=2, Release=0.1, Release_sites=[], NumRelease=10^3,
                                  alpha=0, phi1=1, phi2=1, wa=1, sex=:b, M=4, filename="Sp", homing=:b,
                                  beta=100, track=false, i0=1, stats_only=true, initN=false,
                                  Nfv0=[], Nmv0=[], Nfv=[], Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    namesG = _names_genotypes(M)
    R1Ind  = sum([occursin.(allele[i], namesG) for i in eachindex(allele)]) .> 0
    D1Ind  = sum([occursin.(d_allele[i], namesG) for i in eachindex(d_allele)]) .> 0
    SInds  = Hex_band(; dia=dia)[Int(ceil(dia * sqrt(2 / 3)))]   # sites in the check band

    if isempty(Release_sites)
        Release_sites = [dia]
    end
    f = wmf[3][1, 1]
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B, c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in   = 1 - mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    tr  = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)

    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicateO_cut_flagged(i; threshold=threshold, wmf=wmf, Nmv=Nmv, Nfv=Nfv,
                                   SexDistortion=SexDistortion, e=e, Adj=Adj, maxgen=maxgen,
                                   alpha=alpha, wa=wa, phi1=phi1, phi2=phi2, Release=Release,
                                   Release_sites=Release_sites, NumRelease=NumRelease, M=M,
                                   Model=Model, sex=sex, homing=homing, R1Ind=R1Ind, D1Ind=D1Ind,
                                   SInds=SInds, beta=beta, track=track, fn_prefix=fn_prefix,
                                   stats_only=stats_only)
    end
end


#==============================================================================
  9. GAMETE (INHERITANCE) MATRICES
==============================================================================#

# Collapse an allele-pair matrix into unordered genotype probabilities
function Genot(; A, M2, M)
    v = Array{Float64}(undef, M2)
    k = 0
    for i in 1:M
        v[k+=1] = A[i, i]
        for j in (i+1):M
            v[k+=1] = A[i, j] + A[j, i]
        end
    end
    return v
end

# Table highlighters for the show=true tables. PrettyTables v3 uses TextHighlighter
# (the old keyword form Highlighter(f=..., crayon=...) no longer exists).
# `f(data, i, j)` selects cells; keywords are Crayon options (bold, foreground, background).
function _text_hl(f; kw...)
    c = PrettyTables.Crayon(; kw...)
    return isdefined(PrettyTables, :TextHighlighter) ? PrettyTables.TextHighlighter(f, c) :
                                                       PrettyTables.Highlighter(f, c)
end
_hl_rows(rows) = _text_hl((data, i, j) -> i in rows; background=:light_blue)

# Gametes produced by each genotype (rows) per allele (columns); returns transpose
function E_Make_1(; c=0.95, j=0.03, a=0.01, b=0.001, show=false, M=4, title="Gametes produces")
    if M == 5
        E5 = vcat(
            [1 0 0 0 0],
            [(1-c)/2 (1/2+c*(1-j)*(1-a)/2) c*j*(1-b)/2 c*j*b/2 c*(1-j)*a/2],
            [1/2 0 1/2 0 0],
            [1/2 0 0 1/2 0],
            [(1-c)/2 0 c*j*(1-b)/2 c*j*b/2 (1/2+c*(1-j)/2)],
            [0 1 0 0 0],
            [0 1/2 1/2 0 0],
            [0 1/2 0 1/2 0],
            [0 1/2 0 0 1/2],
            [0 0 1 0 0],
            [0 0 1/2 1/2 0],
            [0 0 1/2 0 1/2],
            [0 0 0 1 0],
            [0 0 0 1/2 1/2],
            [0 0 0 0 1]
        )
        if show
            namesG = ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]
            namesA = ["W", "D", "N", "R", "O"]
            pretty_table(E5, column_labels=namesA, table_format=TextTableFormat(borders=text_table_borders__borderless),
                         row_labels=namesG, highlighters=[_hl_rows([2, 6, 7, 8, 9])], title=title)
        end
    end
    if M == 4
        E5 = vcat(
            [1 0 0 0],
            [(1-c)/2 (1/2+c*(1-j)/2) c*j*(1-b)/2 c*j*b/2],
            [1/2 0 1/2 0],
            [1/2 0 0 1/2],
            [0 1 0 0],
            [0 1/2 1/2 0],
            [0 1/2 0 1/2],
            [0 0 1 0],
            [0 0 1/2 1/2],
            [0 0 0 1]
        )
        if show
            namesG = ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"]
            namesA = ["W", "D", "N", "R"]
            pretty_table(E5, column_labels=namesA, table_format=TextTableFormat(borders=text_table_borders__borderless),
                         row_labels=namesG, highlighters=[_hl_rows([2, 5, 6, 7])], title=title)
        end
    end
    return transpose(E5)
end

# Offspring genotype probabilities for every (female, male) genotype pair:
# e[i][j] = probabilities for female genotype i × male genotype j
function E_mat(; Ef, Em, M2, M)
    E_temp = [[Ef[:, i] * transpose(Em[:, j]) for i in 1:M2] for j in 1:M2]
    e = [[Genot(; A=E_temp[i][j], M=M, M2=M2) for i in 1:M2] for j in 1:M2]
    return e
end

# Sex-specific homing: :b both sexes, :m males only, :f females only
function E_Make(; M=4, homing=:b, c_m=0.8, j_m=0.2, a_m=0.1, b_m=0.1, c_f=0.8, j_f=0.2, a_f=0.1, b_f=0.1, show=false)
    M2 = Int(M * (M + 1) / 2)
    if homing == :b
        show && println("Homing in both sexes")
        Em = E_Make_1(; c=c_m, j=j_m, a=a_m, b=b_m, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=c_f, j=j_f, a=a_f, b=b_f, show=show, M=M, title="Gametes produced (females)")
    elseif homing == :m
        show && println("Homing in males")
        Em = E_Make_1(; c=c_m, j=j_m, a=a_m, b=b_m, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=0, j=0, a=0, b=0, show=false, M=M, title="Gametes produced (females)")
    elseif homing == :f
        show && println("Homing in females")
        Em = E_Make_1(; c=0, j=0, a=0, b=0, show=false, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=c_f, j=j_f, a=a_f, b=b_f, show=show, M=M, title="Gametes produced (females)")
    end
    return E_mat(; Ef=Ef, Em=Em, M2=M2, M=M)
end

# SD with alternative resistance allele
function E_Make_2(; c=0.95, j=0.03, b=0.001, b2=0.5, show=false, M=5, title="Gametes produces")
    if M == 5
        E5 = vcat(
            [1 0 0 0 0],
            [(1-c)/2 (1/2+c*(1-j)/2) c*j*(1-b)/2 c*j*b*(1-b2)/2 c*j*b*b2/2],
            [1/2 0 1/2 0 0],
            [1/2 0 0 1/2 0],
            [1/2 0 0 0 1/2],
            [0 1 0 0 0],
            [0 1/2 1/2 0 0],
            [0 1/2 0 1/2 0],
            [0 1/2 0 0 1/2],
            [0 0 1 0 0],
            [0 0 1/2 1/2 0],
            [0 0 1/2 0 1/2],
            [0 0 0 1 0],
            [0 0 0 1/2 1/2],
            [0 0 0 0 1]
        )
        if show
            namesG = ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]
            namesA = ["W", "D", "N", "R", "O"]
            pretty_table(E5, column_labels=namesA, table_format=TextTableFormat(borders=text_table_borders__borderless),
                         row_labels=namesG, highlighters=[_hl_rows([2, 6, 7, 8, 9])], title=title)
        end
    end
    if M == 4
        E5 = vcat(
            [1 0 0 0],
            [(1-c)/2 (1/2+c*(1-j)/2) c*j*(1-b)/2 c*j*b/2],
            [1/2 0 1/2 0],
            [1/2 0 0 1/2],
            [0 1 0 0],
            [0 1/2 1/2 0],
            [0 1/2 0 1/2],
            [0 0 1 0],
            [0 0 1/2 1/2],
            [0 0 0 1]
        )
        if show
            namesG = ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"]
            namesA = ["W", "D", "N", "R"]
            pretty_table(E5, column_labels=namesA, table_format=TextTableFormat(borders=text_table_borders__borderless),
                         row_labels=namesG, highlighters=[_hl_rows([2, 5, 6, 7])], title=title)
        end
    end
    return transpose(E5)
end

function E_Make2(; M=4, homing=:b, c_m=0.8, j_m=0.2, b_m=0.1, c_f=0.8, j_f=0.2, b2=0.5, b_f=0.1, show=false)
    M2 = Int(M * (M + 1) / 2)
    if homing == :b
        show && println("Homing in both sexes")
        Em = E_Make_2(; c=c_m, j=j_m, b=b_m, b2=b2, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_2(; c=c_f, j=j_f, b=b_f, b2=b2, show=show, M=M, title="Gametes produced (females)")
    elseif homing == :m
        show && println("Homing in males")
        Em = E_Make_2(; c=c_m, j=j_m, b=b_m, b2=b2, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_2(; c=0, j=0, b=0, b2=b2, show=false, M=M, title="Gametes produced (females)")
    elseif homing == :f
        show && println("Homing in females")
        Em = E_Make_2(; c=0, j=0, b=0, b2=b2, show=false, M=M, title="Gametes produced (males)")
        Ef = E_Make_2(; c=c_f, j=j_f, b2=b2, b=b_f, show=show, M=M, title="Gametes produced (females)")
    end
    return E_mat(; Ef=Ef, Em=Em, M2=M2, M=M)
end

# Multiplexing: haplotype transition probabilities
function Transition1(; c=0.95, j=0.03, b=0.001, show=false, M=3)
    M0 = Int(M * (M + 1) / 2)   # non-drive haplotypes
    M1 = M0 + 1                 # all haplotypes
    M2 = Int(M1 * (M1 + 1) / 2) # all pairs of haplotypes
    MM_Ind = collect(zip(reduce(vcat, [1:i for i in 1:M0]), reduce(vcat, [repeat([i], i) for i in 1:M0])))

    Tr = Array{Array{Float64}}(undef, M2)
    for ii in eachindex(MM_Ind)
        Tr[ii] = zeros(M1)
        Tr[ii][MM_Ind[ii][1]] += 1/2
        Tr[ii][MM_Ind[ii][2]] += 1/2
    end
    # local positional variant of Genot
    genot_local = function (A; M, M2)
        v = Array{Float64}(undef, M2)
        k = 0
        for i in 1:M
            v[k+=1] = A[i, i]
            for j in (i+1):M
                v[k+=1] = A[i, j] + A[j, i]
            end
        end
        return v
    end

    M_Ind = [(i, j) for i in 1:M for j in i:M]
    pnd = ones(M1) ./ 2                            # prob. of no drive
    pnd[1:M] = pnd[1:M] * (1 - c * (1 - j))
    pnd[1] = pnd[1] * (1 - c * (1 - j))
    for ll in 1:M0
        # gamete formation with D (re-weighted)
        v = [[(1-c)/2, c*j*(1-b)/2, c*j*b/2] ./ sum([(1-c)/2, c*j*(1-b)/2, c*j*b/2]), [0, 1, 0], [0, 0, 1]]
        vv = deepcopy(genot_local(v[first(M_Ind[ll])] * transpose(v[last(M_Ind[ll])]), M=M, M2=M0))
        Tr[length(MM_Ind)+ll] = deepcopy(vcat(pnd[ll] * vv, 1 - pnd[ll]))
    end
    Tr[M2] = zeros(M1)
    Tr[M2][M1] = 1
    return Tr
end

function E_Make_Multi(; M=3, homing=:b, c_m=0.95, j_m=0.03, b_m=0.001, c_f=0.95, j_f=0.03, b_f=0.001, show=false)
    M1 = Int(M * (M + 1) / 2) + 1
    M2 = Int(M1 * (M1 + 1) / 2)
    MM_Ind = collect(zip(reduce(vcat, [1:i for i in 1:M1]), reduce(vcat, [repeat([i], i) for i in 1:M1])))

    if homing == :b
        show && println("Homing in both sexes")
        Em = Transition1(; c=c_m, j=j_m, b=b_m, show=show, M=M)
        Ef = Transition1(; c=c_f, j=j_f, b=b_f, show=show, M=M)
    elseif homing == :m
        show && println("Homing in males")
        Em = Transition1(; c=c_m, j=j_m, b=b_m, show=show, M=M)
        Ef = Transition1(; c=0, j=0, b=0, show=false, M=M)
    elseif homing == :f
        show && println("Homing in females")
        Em = Transition1(; c=0, j=0, b=0, show=false, M=M)
        Ef = Transition1(; c=c_f, j=j_f, b=b_f, show=show, M=M)
    end

    E_temp = [[Ef[i] * transpose(Em[j]) for j in 1:M2] for i in 1:M2]
    for j in 1:M2, i in 1:M2, d in 1:M1
        E_temp[i][j][d, d] = E_temp[i][j][d, d] / 2
    end
    e = [[[E_temp[i][j][MM_Ind[ii][1], MM_Ind[ii][2]] + E_temp[i][j][MM_Ind[ii][2], MM_Ind[ii][1]]
           for ii in eachindex(MM_Ind)] for i in 1:M2] for j in 1:M2]
    return e
end

# ----- Multiplexing 2 gametes --------------------------------------------------

# Gametes when the partner haplotype of `g` is DD (drive). Sets entries of
# `gametes` in place.
function _gametes_DD_partner!(gametes, g, Names2, DO_vec; k, c1, c2, j1, j2, b1, l, delta1, delta)
    setg!(name, val) = (gametes[findall(==(name), Names2)] .= val)
    a1, a2 = string(g[1]), string(g[2])
    if a1 == "W" && a2 != "W"               # cut at locus 1 only
        setg!(g, 1/2 * (1 - c1))
        gametes[indexin(string.(a2, ["N", "R"]), Names2)] = [1/2*c1*j1*(1-b1), 1/2*c1*j1*b1]
        gametes[DO_vec] = 1/2*c1*(1-j1) .* delta1 .+ [1/2, 0, 0, 0]
    elseif a1 != "W" && a2 == "W"           # cut at locus 2 only
        setg!(g, 1/2 * (1 - c1))
        gametes[indexin(string.(a1, ["N", "R"]), Names2)] = [1/2*c1*j1*(1-b1), 1/2*c1*j1*b1]
        gametes[DO_vec] = 1/2*c1*(1-j1) .* delta1 .+ [1/2, 0, 0, 0]
    elseif a1 == "W" && a2 == "W"           # both loci wild type
        setg!("WW", 1/2*(1-c2))
        setg!("NN", 1/2*c2*k*j2 + 1/2*c2*(1-k)*c1*j1^2*(1-b1)^2)
        setg!("RR", 1/2*c2*(1-k)*c1*j1^2*b1^2)
        setg!("WN", 1/2*c2*(1-k)*j1*(1-b1)*(1-c1)*(1-l))
        setg!("WR", 1/2*c2*(1-k)*j1*b1*(1-c1)*(1-l))
        setg!("NW", 1/2*c2*(1-k)*j1*(1-b1)*(1-c1)*l)
        setg!("RW", 1/2*c2*(1-k)*j1*b1*(1-c1)*l)
        setg!("NR", 1/2*c2*(1-k)*j1^2*b1*c1*(1-b1))
        setg!("RN", 1/2*c2*(1-k)*j1^2*b1*c1*(1-b1))
        gametes[DO_vec] .= [1/2, 0, 0, 0] .+ 1/2 .* c2*k*(1-j2) .* delta .+
                           1/2*c2*(1-k) .* delta1 .* (1 - j1 + j1*c1*(1-j1))
    end
    return gametes
end

# Gamete distribution (over the 13 haplotypes in Names2) of one genotype "AB||CD"
function Gametes(Gen; k=0.5, c1=0.995, c2=0.995, c11=100, j1=0.035, j2=0.035, b1=0.001, l=0.5,
                 QFS=2, show_input=false, show=false, delta1, delta, deltaA=0.0009, deltaB=0.0032)
    Names2 = _multi2_names()[1]
    g1, g2 = Gen[1:2], Gen[5:6]            # the two haplotypes
    # NOTE (unchanged): `1 < c11 < 0` is never true, so c11 is not used here
    if 1 < c11 < 0
        c11 = 1 - (1 - c1)^2
    end
    gametes = zeros(length(Names2))
    if show
        println(g1)
        println(g2)
    end
    DO_vec = [findall(==(g), Names2)[1] for g in ["DD", "DO", "OD", "OO"]]
    ddkw = (k=k, c1=c1, c2=c2, j1=j1, j2=j2, b1=b1, l=l, delta1=delta1, delta=delta)

    # Mendelian: each haplotype passed on with prob 1/2 (1 if homozygous and hom=true)
    function mendel!(hom)
        i1, i2 = findall(==(g1), Names2), findall(==(g2), Names2)
        gametes[i1] .= 1/2
        gametes[i2] .= 1/2
        hom && i1 == i2 && (gametes[i1] .= 1)
    end
    # One drive locus (OD or DO) opposite a wild-type allele at that locus
    function single_drive!(X, idx, dAB)
        gametes[indexin(X, Names2)] = [(1-c1)/2, 1/2*c1*j1*(1-b1), 1/2*c1*j1*b1]
        gametes[DO_vec[idx]] = [1 - dAB, dAB] .* c1*(1-j1) ./ 2 .+ [1/2, 0]
    end

    if count("D", g1) + count("D", g2) == 0 || count("W", g1) + count("W", g2) == 0
        mendel!(true)                                   # no drive, or nothing to cut
    elseif g2 == "OD" && g1[2] == 'W'
        single_drive!(string.(g1[1], ["W", "N", "R"]), [3, 4], deltaB)
    elseif g1 == "OD" && g2[2] == 'W'
        single_drive!(string.(g2[1], ["W", "N", "R"]), [3, 4], deltaB)
    elseif g2 == "DO" && g1[1] == 'W'
        single_drive!(string.(["W", "N", "R"], g1[2]), [2, 4], deltaA)
    elseif g1 == "DO" && g2[1] == 'W'
        single_drive!(string.(["W", "N", "R"], g2[2]), [2, 4], deltaA)
    elseif (g2 == "DO" && g1[1] != 'W') || (g1 == "DO" && g2[1] != 'W')
        mendel!(false)
    elseif (g2 == "OD" && g1[2] != 'W') || (g1 == "OD" && g2[2] != 'W')
        mendel!(false)
    elseif count("D", g1) == 2
        _gametes_DD_partner!(gametes, g2, Names2, DO_vec; ddkw...)
    end
    # Partner of a DD haplotype in second position. (The original also had this
    # inside the chain above; running it once here gives the same result.)
    if count("D", g2) == 2
        _gametes_DD_partner!(gametes, g1, Names2, DO_vec; ddkw...)
    end
    return gametes
end

# Multiplexing 2: offspring genotype probabilities for every (female, male) pair
# of the 91 two-locus genotypes.
function Gametes_all(; k=0.75, c1=0.995, c2=0.995, c11=100, j1=0.035, j2=0.035, b1=0.001,
                     delta1=[1, 0, 0, 0], delta=[1, 0, 0, 0], deltaA=0.0009, deltaB=0.0032, l=0.5,
                     useQFS=true, QFS=2, showF=false, showM=false, homing=:b, precision=4)
    Names2, Haplotypes, MM_Ind = _multi2_names()

    # Preset cleavage-outcome data sets (QFS2 / QFS5) override the delta inputs
    if useQFS
        if QFS == 2
            deltaA = 0.0009
            deltaB = 0.0032
            delta1 = [1 - 0.024, 0.024, 0, 0]
            delta  = delta1
        elseif QFS == 5
            deltaA = 0.0009
            deltaB = 0.0032
            delta1 = [0.9483, 0.0032, 0.0009, 0.0476]
            delta  = delta1
        end
    end

    # NOTE (unchanged): the condition tests c1 > 1; c11 > 1 may have been intended
    if (c11 < 0) || (c1 > 1)
        c11 = 1 - (1 - c1)^2
    end

    M  = 13
    M2 = Int(M * (M + 1) / 2)   # 91

    # Gamete distribution of every genotype, for males (Em) and females (Ef)
    gam(c1_, c11_, c2_) = Gametes.(Haplotypes; k=k, c1=c1_, c11=c11_, c2=c2_, j1=j1, j2=j2, b1=b1,
                                   delta1=delta1, delta=delta, deltaA=deltaA, deltaB=deltaB, l=l,
                                   show=false)
    if homing == :b
        Em = gam(c1, c11, c2)
        Ef = gam(c1, c11, c2)
    elseif homing == :m
        println("Homing in males")
        Em = gam(c1, c11, c2)
        Ef = gam(0, 0, 0)
    elseif homing == :f
        println("Homing in females")
        Em = gam(0, 0, 0)
        Ef = gam(c1, c11, c2)
    end

    # Female × male gamete pairs → unordered offspring genotypes
    E_temp = [[Ef[i] * transpose(Em[j]) for i in 1:M2] for j in 1:M2]
    for j in 1:M2, i in 1:M2, d in 1:M
        E_temp[i][j][d, d] = E_temp[i][j][d, d] / 2   # diagonal is added twice below
    end
    e = [[[E_temp[i][j][MM_Ind[ii][1], MM_Ind[ii][2]] + E_temp[i][j][MM_Ind[ii][2], MM_Ind[ii][1]]
           for ii in eachindex(MM_Ind)] for i in 1:M2] for j in 1:M2]

    # Optional tables: genotypes whose gametes are not simple Mendelian (0, 1/2, 1)
    for (doshow, Eg, title) in ((showF, Ef, "Gametes produced (females)"),
                                (showM, Em, "Gametes produced (males)"))
        doshow || continue
        exc = findall(y -> length(y) < M, [findall(x -> x in (0, 0.5, 1), gg) for gg in Eg])
        namesG2 = Haplotypes[exc]
        hl_lo = _text_hl((data, i, j) -> 0 < data[i, j] < 0.5; bold=true, foreground=:red)
        hl_hi = _text_hl((data, i, j) -> 0.5 < data[i, j] < 1; bold=true, foreground=:green)
        E2 = reshape([Eg[exc[i]][j] for j in eachindex(Names2) for i in eachindex(namesG2)],
                     length(namesG2), length(Names2))
        pretty_table(round.(E2, digits=precision), column_labels=Names2,
                     table_format=TextTableFormat(borders=text_table_borders__borderless),
                     row_labels=namesG2, highlighters=[hl_lo, hl_hi], title=title)
    end
    return e
end


#==============================================================================
  10. FITNESS INPUTS
  Return (wm, wf, ff): male viability, female viability, pair fecundity.
==============================================================================#

function Fitness_input1(; sigma=0, s=1, h=1, hN=0, hDR=0, hRO=0, show=true, M=4, hm=0, hmO=0, f=12,
                        inter=true, wf_nb=[], non_func=["D", "N", "O"])
    M2 = Int(M * (M + 1) / 2)
    wm = ones(M2)
    if M == 5
        wf0 = [1, 1-h*s, 1-hN*s, 1-sigma, 1-h*s, 1-s, 1-s, (1-sigma)*(1-hDR*s), 1-s, 1-s,
               (1-sigma)*(1-hN*s), 1-s, (1-sigma)^2, (1-sigma)*(1-hRO*s), 1-s]
        wfth = ["1", "1-h*s", "1-hN*s", "1-sigma", "1-h*s", "1-s", "1-s", "(1- sigma)(1-hDR*s)", "1-s", "1-s",
                "(1-sigma)(1-hN*s)", "1-s", "(1-sigma)^2", "(1-sigma)(1-hRO*s)", "1-s"]
        namesG = ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]
    end
    if M == 4
        wf0 = [1, 1-h*s, 1-hN*s, 1-sigma, 1-s, 1-s, (1-sigma)*(1-hDR*s), 1-s, (1-sigma)*(1-hN*s), (1-sigma)^2]
        wfth = ["1", "1-h*s", "1-h_N*s", "1-sigma", "1-s", "1-s", "(1-sigma)(1-hDR*s)", "1-s",
                "(1-sigma)(1-hN*sN)", "(1-sigma)^2"]
        namesG = ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"]
    end
    namesF = ["Males", "Females", "Formula (female)"]

    # Female viability is 0/1 (sterile or not); fitness effects go into fecundity ff
    if inter
        wf = ones(M2)
        wf[wf0 .<= 0] .= 0
    else
        wf = wf_nb
    end
    wm[occursin.("D", namesG)] .= 1 - hm
    wm[occursin.("O", namesG)] .= 1 - hmO

    ff = ones(M2, M2) #.* f
    for j in 1:M2, i in 1:M2
        ff[i, j] = wf0[i] * wm[j] * f
    end
    if show
        pretty_table([wm wf0 wfth], column_labels=namesF, table_format=TextTableFormat(borders=text_table_borders__borderless),
                     row_labels=namesG, alignment=:l)
    end
    return wm, wf, ff
end

function Fitness_input1_old(; sigma=0.02, s=1, h=1, hN=0.03, hDR=0.03, show=true, M=4)
    M2 = Int(M * (M + 1) / 2)
    wm = ones(M2)
    if M == 5
        wf = [1, 1-h*s, 1-hN*s, 1-sigma, 1-h*s, 1-s, 1-s, (1-sigma)*(1-hDR*s), 1-s, 1-s,
              (1-sigma)*(1-hN*s), 1-s, (1-sigma)^2, (1-sigma)*(1-hDR*s), 1-s]
        wfth = ["1", "1-hs", "1-hN*s", "1-sigma", "1-h*s", "1-s", "1-s", "(1- sigma)(1-hDRs)", "1-s", "1-s",
                "(1-sigma)(1-hN*s)", "1-s", "(1-sigma)^2", "(1-sigma)(1-hDR*s)", "1-s"]
        namesG = ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]
    end
    if M == 4
        wf = [1, 1-h*s, 1-hN*s, 1-sigma, 1-s, 1-s, (1-sigma)*(1-hDR*s), 1-s, (1-sigma)*(1-hN*s), (1-sigma)^2]
        wfth = ["1", "1-h*s", "1-h_N*s", "1-sigma", "1-s", "1-s", "(1-sigma)(1-hDR*s)", "1-s",
                "(1-sigma)(1-hN*sN)", "(1-sigma)^2"]
        namesG = ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"]
    end
    namesF = ["Males", "Females", "Formula (female)"]
    if show
        pretty_table([round.(Int64, ones(M2)) wf wfth], column_labels=namesF,
                     table_format=TextTableFormat(borders=text_table_borders__borderless), row_labels=namesG, alignment=:l)
    end
    return wm, wf
end

function Fitness_input2(; sigma=0.02, s=1, h=1, hN=0.03, hDR=0.03, show=true, M=4, hm=0, f=12, sex=:f,
                        non_func=["D", "N", "O"])
    M2 = Int(M * (M + 1) / 2)
    if M == 5
        wf0 = [1, 1-h*s, 1-hN*s, 1-sigma, 1-sigma, 1-s, 1-s, (1-sigma)*(1-hDR*s),
               (1-sigma)*(1-hN*s),   # in SDMD ???
               1-s, (1-sigma)*(1-hN*s), (1-sigma)*(1-hN*s), (1-sigma)^2, (1-sigma)^2, (1-sigma)^2]
        wfth = ["1", "1-hs", "1-hN*s", "1-sigma", "1-h*s", "1-s", "1-s", "(1- sigma)(1-hDRs)", "1-s", "1-s",
                "(1-sigma)(1-hN*s)", "1-s", "(1-sigma)^2", "(1-sigma)(1-hDR*s)", "1-s"]
        namesG = ["WW", "WD", "WN", "WR", "WO", "DD", "DN", "DR", "DO", "NN", "NR", "NO", "RR", "RO", "OO"]
    end
    if M == 4
        wf0 = [1, 1-h*s, 1-hN*s, 1-sigma, 1-s, 1-s, (1-sigma)*(1-hDR*s), 1-s, (1-sigma)*(1-hN*s), (1-sigma)^2]
        wfth = ["1", "1-h*s", "1-h_N*s", "1-sigma", "1-s", "1-s", "(1-sigma)(1-hDR*s)", "1-s",
                "(1-sigma)(1-hN*sN)", "(1-sigma)^2"]
        namesG = ["WW", "WD", "WN", "WR", "DD", "DN", "DR", "NN", "NR", "RR"]
    end
    if sex == :f
        wm = ones(M2)                          # FIX: was `wm0 = ...`, leaving wm undefined
        wm[occursin.("D", namesG)] .= 1 - hm
    elseif sex == :b
        wm = wf0
    else
        throw(ArgumentError("Fitness costs in (:f, :b) only"))
    end
    namesF = ["Males", "Females", "Formula (female)"]
    wf = ones(M2)
    wf[wf0 .<= 0] .= 0

    ff = ones(M2, M2) .* f
    for j in 1:M2, i in 1:M2
        ff[i, j] = wf0[i] * wm[j] * f
    end
    if show
        pretty_table([wm wf0 wfth], column_labels=namesF, table_format=TextTableFormat(borders=text_table_borders__borderless),
                     row_labels=namesG, alignment=:l)
    end
    return wm, wf, ff
end

function FitnessMulti2(; sigma=0, s=1, h=0, hDR=0, hN=0, sN=1, hm=0, F=12,
                       Hnames=vcat(["WW", "WN", "WR", "NN", "NR", "RR"], "DD"),
                       showFormula=false, showValues=false)
    Hn = length(Hnames)
    MM_Ind = collect(zip(reduce(vcat, [1:i for i in 1:Hn]), reduce(vcat, [repeat([i], i) for i in 1:Hn])))

    Nind  = findall(>(0), count.("N", Hnames))
    R1ind = [count("N", Hnames[i]) < 1 ? count("R", Hnames[i]) : 0 for i in 1:Hn]
    Rind  = findall(>(0), R1ind)
    Rind2 = findall(>(1), R1ind)
    Dind  = [Hn]
    f = ones(Hn, Hn)
    text_females = string.(ones(Hn, Hn))
    for i in 1:Hn
        for j in i:Hn
            if (i in Nind) + (j in Nind) > 1
                f[i, j] = 1 - s; text_females[i, j] = "1-s"
            elseif (i in Nind) + (j in Nind) == 1
                if j in Rind
                    f[i, j] = (1 - s*hN) * (1 - sigma)^R1ind[j]; text_females[i, j] = "(1-s*hN)(1-sigma)^$(R1ind[j])"
                elseif i in Rind
                    f[i, j] = (1 - s*hN) * (1 - sigma)^R1ind[i]; text_females[i, j] = "(1-s*hN)(1-sigma)^$(R1ind[i])"
                elseif j in Dind
                    f[i, j] = (1 - s); text_females[i, j] = "(1-s)"
                else
                    f[i, j] = (1 - hN); text_females[i, j] = "(1-hN)"
                end
            elseif (i in Nind) + (j in Nind) < 1
                if (i in Rind) + (j in Rind) > 1
                    f[i, j] = (1 - sigma)^(R1ind[j] + R1ind[i]); text_females[i, j] = "(1-sigma)^$(R1ind[j]+R1ind[i])"
                elseif (i in Rind) + (j in Rind) == 1
                    if j in Dind
                        f[i, j] = (1 - sigma)^(R1ind[i]) * (1 - h); text_females[i, j] = "(1-h)(1-sigma)^$(R1ind[i])"
                    else
                        f[i, j] = (1 - sigma)^(R1ind[j]); text_females[i, j] = "(1-sigma)^$(R1ind[j])"
                    end
                elseif (i in Rind) + (j in Rind) < 1
                    if (i in Dind) + (j in Dind) == 1
                        f[i, j] = (1 - h); text_females[i, j] = "(1-h)"
                    elseif (i in Dind) + (j in Dind) > 1
                        f[i, j] = (1 - s); text_females[i, j] = "(1-s)"
                    end
                end
            end
            if (i in Rind2) + (j in Dind) == 2
                f[i, j] = (1 - hDR) * (1 - sigma^2); text_females[i, j] = "(1-hDR)*(1-sigma^2)"
            end
        end
    end
    if showFormula
        for i in 1:Hn, j in i:Hn
            text_females[j, i] = text_females[i, j]
        end
        pretty_table(text_females, column_labels=Hnames, table_format=TextTableFormat(borders=text_table_borders__borderless),
                     row_labels=Hnames, alignment=:l)
    end
    if showValues
        for i in 1:Hn, j in i:Hn
            f[j, i] = f[i, j]
        end
        pretty_table(f, column_labels=Hnames, table_format=TextTableFormat(borders=text_table_borders__borderless),
                     row_labels=Hnames, alignment=:l)
    end
    ffit = [f[MM_Ind[i][1], MM_Ind[i][2]] for i in eachindex(MM_Ind)]

    wffit = ones(length(MM_Ind))
    wffit[ffit .<= 0] .= 0

    mfit = ones(length(MM_Ind))
    mfit[length(MM_Ind)-Hn+1:end] .= 1 - hm     # male fitness costs

    ffiteggs = ones(length(MM_Ind), length(MM_Ind))
    for i in 1:length(MM_Ind), j in 1:length(MM_Ind)
        ffiteggs[i, j] = ffit[i] * mfit[j] * F
    end
    return mfit, wffit, ffiteggs
end

# Multiplexing 2 fitness: (male viability, female viability 0/1, pair fecundity)
# over the 91 genotypes built from haplotypes Hnames.
function FitnessMulti22(; sigma=0, s=1, h=0, hDR=0, hN=0, sN=1, hm=0, F=12,
                        Hnames=_multi2_names()[1], showFormula=false, showValues=false,
                        showSummary=false)
    Hn = length(Hnames)
    # pair of haplotypes → genotype index
    MM_Ind = collect(zip(reduce(vcat, [1:i for i in 1:Hn]), reduce(vcat, [repeat([i], i) for i in 1:Hn])))

    Nind  = findall(>(0), count.("N", Hnames) .+ count.("OO", Hnames))   # non-functional
    R1ind = [count("N", Hnames[i]) < 1 ? count("R", Hnames[i]) : 0 for i in 1:Hn]
    Rind  = findall(>(0), R1ind)                                         # resistant
    Rind2 = findall(>(1), R1ind)
    Dind  = findall(>(0), count.("D", Hnames))                           # drive
    Dind2 = findall(>(1), count.("D", Hnames))
    Wind2 = findall(>(1), count.("W", Hnames) .+ count.("R", Hnames))   # two functional copies

    # Female fitness f[i, j] for haplotypes i ≤ j (upper triangle) + formula text
    f = ones(Hn, Hn)
    text_females = string.(ones(Hn, Hn))
    for i in 1:Hn
        for j in i:Hn
            if (i in Nind) + (j in Nind) > 1
                f[i, j] = 1 - s; text_females[i, j] = "1-s"
            elseif (i in Nind) + (j in Nind) == 1
                if j in Rind
                    f[i, j] = (1 - s*hN) * (1 - sigma)^R1ind[j]; text_females[i, j] = "(1-hN*s)(1-sigma)^$(R1ind[j])"
                elseif i in Rind
                    f[i, j] = (1 - s*hN) * (1 - sigma)^R1ind[i]; text_females[i, j] = "(1-hN*s)(1-sigma)^$(R1ind[i])"
                elseif j in Dind
                    f[i, j] = 1 - s; text_females[i, j] = "1-s"
                else
                    # NOTE (unchanged): value is 1-hN but the label says 1-hN*s (same when s = 1)
                    f[i, j] = 1 - hN; text_females[i, j] = "1-hN*s"
                end
            elseif (i in Nind) + (j in Nind) < 1
                if (i in Rind) + (j in Rind) > 1
                    f[i, j] = (1 - sigma)^(R1ind[j] + R1ind[i]); text_females[i, j] = "(1-sigma)^$(R1ind[j]+R1ind[i])"
                elseif (i in Rind) + (j in Rind) == 1
                    if j in Dind
                        # NOTE (unchanged): value uses (1-h), label says (1-h*s)
                        f[i, j] = (1 - sigma)^(R1ind[i]) * (1 - h); text_females[i, j] = "(1-h*s)(1-sigma)^$(R1ind[i])"
                    else
                        f[i, j] = (1 - sigma)^(R1ind[j]); text_females[i, j] = "(1-sigma)^$(R1ind[j])"
                    end
                elseif (i in Rind) + (j in Rind) < 1
                    if (i in Dind) + (j in Dind) == 1
                        f[i, j] = 1 - h*s; text_females[i, j] = "1-h*s"
                    elseif (i in Dind) + (j in Dind) > 1
                        f[i, j] = 1 - s; text_females[i, j] = "1-s"
                    end
                end
            end
            if (i in Rind2) + (j in Dind) == 2
                # NOTE (unchanged): value uses (1-sigma^2), label says (1-sigma)^2
                f[i, j] = (1 - hDR) * (1 - sigma^2); text_females[i, j] = "(1-hDR)*(1-sigma)^2"
            end
            # A drive haplotype without a resistant or two-functional-copy partner → sterile
            if (i in Dind) + (j in Dind) > 0
                if ((i in Rind) + (j in Rind) == 0) & ((i in Wind2) + (j in Wind2) == 0)
                    f[i, j] = 1 - s; text_females[i, j] = "1-s"
                end
            end
        end
    end

    hl_zero = _text_hl((data, i, j) -> data[i, j] <= 0; bold=true, background=196)
    hl_one  = _text_hl((data, i, j) -> data[i, j] >= 1; bold=true, background=:blue)
    tf = TextTableFormat(borders=text_table_borders__borderless)
    if showFormula
        for i in 1:Hn, j in i:Hn
            text_females[j, i] = text_females[i, j]
        end
        pretty_table(text_females, column_labels=Hnames, table_format=tf, row_labels=Hnames, alignment=:l)
    end
    if showValues
        for i in 1:Hn, j in i:Hn
            f[j, i] = f[i, j]     # mirror to lower triangle (only affects display)
        end
        pretty_table(f, column_labels=Hnames, table_format=tf, highlighters=[hl_zero, hl_one],
                     title="Female fitness coefficients", row_labels=Hnames, alignment=:c)
    end

    ffit  = [f[MM_Ind[i][1], MM_Ind[i][2]] for i in eachindex(MM_Ind)]
    wffit = ones(length(MM_Ind))
    wffit[ffit .<= 0] .= 0
    mfit  = ones(length(MM_Ind))            # male costs (hm) currently switched off

    ffiteggs = ones(length(MM_Ind), length(MM_Ind))
    for i in 1:length(MM_Ind), j in 1:length(MM_Ind)
        ffiteggs[i, j] = ffit[i] * mfit[j] * F
    end

    if showSummary
        names0  = ["WW", "WR", "RR", "WN", "NR", "NN", "OD", "OO", "DD"]
        names02 = ["WW", "WR/RW", "RR", "WN/NW", "NR/RN", "NN", "OD/DO", "OO", "DD"]
        inds = indexin(names0, Hnames)
        for i in 1:Hn, j in i:Hn
            f[j, i] = f[i, j]
        end
        pretty_table(f[inds, inds], column_labels=names02, title="Female fitness coefficients",
                     table_format=tf, highlighters=[hl_zero, hl_one], row_labels=names02, alignment=:l)
    end
    return mfit, wffit, ffiteggs
end


#==============================================================================
  11. INITIAL POPULATIONS
==============================================================================#

function Initial(; NN, Nrelease, M2=10, ReleaseGT=[2], tp=Int64)
    Nf0 = zeros(tp, M2)
    Nm0 = zeros(tp, M2)
    Nf0[1] = round(tp, NN * 0.5)
    Nm0[1] = round(tp, NN * 0.5)
    Nm0[ReleaseGT] .= Nrelease          # released males
    return Nf0, Nm0
end

# Spatial: release a fraction pm of each release site's population
function Initial_vec(; N, Nsites, Release_sites, pm=0.1, Rerelease=false, M=4, tp=Int64)
    M2 = Int(M * (M + 1) / 2)
    NN = length(N) < Nsites ? repeat([round(Int, N / Nsites)], Nsites) : N
    Rmv1 = zeros(Nsites)
    Nrelease = round.(tp, NN .* pm .* 0.5)
    Rmv1[Release_sites] = Nrelease[Release_sites]
    Nmv0 = [Initial(; NN=NN[i], Nrelease=Rmv1[i], M2=M2, ReleaseGT=M+1, tp=tp)[2] for i in 1:Nsites]
    Nfv0 = [Initial(; NN=NN[i], Nrelease=0, M2=M2, ReleaseGT=M+1, tp=tp)[1] for i in 1:Nsites]
    Nmv = reshape(reduce(vcat, Nmv0), M2, Nsites)
    Nfv = reshape(reduce(vcat, Nfv0), M2, Nsites)
    return Nfv, Nmv
end

# Spatial: fixed total number released, divided among the release sites
function Initial_vec_fixed(; N, Nsites, Release_sites, pm=0.1, NumRelease=0, ReleaseGT=[2], M2=10, tp=Int64)
    NN = length(N) < Nsites ? repeat([round(tp, N / Nsites)], Nsites) : N
    Rmv1 = zeros(Nsites)
    if NumRelease <= 0
        Nrelease = round.(tp, NN .* pm .* 0.5)
    else
        Nrelease = [round(tp, NumRelease / length(Release_sites)) for n in 1:Nsites]
    end
    Rmv1[Release_sites] = Nrelease[Release_sites]
    Nmv0 = [Initial(; NN=NN[i], Nrelease=Rmv1[i], M2=M2, ReleaseGT=ReleaseGT, tp=tp)[2] for i in 1:Nsites]
    Nfv0 = [Initial(; NN=NN[i], Nrelease=0, M2=M2, ReleaseGT=ReleaseGT, tp=tp)[1] for i in 1:Nsites]
    Nmv = reshape(reduce(vcat, Nmv0), M2, Nsites)
    Nfv = reshape(reduce(vcat, Nfv0), M2, Nsites)
    return Nfv, Nmv
end

# Add a later release to existing populations
function Initial_vec_fixed_delay(; Nfv, Nmv, Release_sites, NumRelease=1000, ReleaseGT=[2])
    Nfv_new = Nfv
    Nmv_new = copy(Nmv)
    for s in Release_sites, g in ReleaseGT
        Nmv_new[g, s] = Nmv_new[g, s] + NumRelease
    end
    return Nfv_new, Nmv_new
end

# Multiplexing 1 initial population
function InputMulti(Npop; M=3, pm=0.1, tp=Int64)
    M1 = Int(M * (M + 1) / 2) + 1
    M2 = Int(M1 * (M1 + 1) / 2)
    Nvec_f = zeros(tp, M2); Nvec_f[1] = round(tp, Npop / 2)
    Nvec_m = zeros(tp, M2); Nvec_m[1] = round(tp, (1 - pm) * Npop / 2)
    Nvec_m[M2] = round(tp, pm * Npop / 2)
    return Nvec_f, Nvec_m
end

# Multiplexing 2 initial population: wild type plus released DD||DD males
function InputMulti2(Npop; M=13, pm=0.1, namesH=_multi2_names()[2], tp=Int64)
    M2 = Int(M * (M + 1) / 2)
    Nvec_f = zeros(tp, M2); Nvec_f[1] = round(tp, Npop / 2)
    Nvec_m = zeros(tp, M2); Nvec_m[1] = round(tp, (1 - pm) * Npop / 2)
    Nvec_m[findall(x -> x == "DD||DD", namesH)] .= round(tp, pm * Npop / 2)
    return Nvec_f, Nvec_m
end


#==============================================================================
  12. SPATIAL GRID (hexagonal lattice)
  Hexagon1 and Hexagon11 list the sites in the same order (row 0, then +1/-1,
  +2/-2, ...), so site indices from Hex_band match the model's sites.
==============================================================================#

# Hexagon of radius n (n+1 sites along each edge). Returns
# (polygons, vertex lists, number of sites, site centres d, zip(x, y)).
# d is an N×2 Float64 matrix: d[:, 1] = x, d[:, 2] = y (was a DataFrame).
# Row spacing 1.5; used for adjacency and plotting.
function Hexagon1(; n)
    zx = [-1, -1, 0, 1, 1, 0]
    zy = [-1 / 2, 1 / 2, 1, 1 / 2, -1 / 2, -1]
    x = [-2*n:2:2*n;]
    y = [0 for xx in x]
    for i in 1:n
        x = vcat(x, Vector([(-2*n+i):2:(2*n-i);]))
        y = vcat(y, [2 * i - i / 2 for j in 1:(2*n-i+1)])
        x = vcat(x, Vector([(-2*n+i):2:(2*n-i);]))
        y = vcat(y, [-2 * i + i / 2 for j in 1:(2*n-i+1)])
    end
    d = hcat(Float64.(x), Float64.(y))   # same values Matrix(DataFrame(x=x, y=y)) gave
    N = length(x)
    v = [[(zx[j] + x[i], zy[j] + y[i]) for j in 1:6] for i in 1:N]
    h_list = [Polygon(GeometryBasics.Point2f[points for points in v[i]]) for i in 1:N]
    c = zip(x, y)
    return h_list, v, N, d, c
end

# Same grid with true hexagon geometry (row spacing √3·d0); d is an N×2 matrix.
function Hexagon11(; n, d0=1)
    zx = [-1, -1, 0, 1, 1, 0] .* d0
    zy = [-sqrt(3) / 3, sqrt(3) / 3, 2 * sqrt(3) / 3, sqrt(3) / 3, -sqrt(3) / 3, -2 * sqrt(3) / 3] .* d0
    x = [-2*n:2:2*n;] .* d0
    y = [0 for xx in x]
    for i in 1:n
        x = vcat(x, Vector([(-2*n+i):2:(2*n-i);] .* d0))
        y = vcat(y, [sqrt(3) * i for j in 1:(2*n-i+1)] .* d0)
        x = vcat(x, Vector([(-2*n+i):2:(2*n-i);] .* d0))
        y = vcat(y, [-sqrt(3) * i for j in 1:(2*n-i+1)] .* d0)
    end
    d = ones(length(x), 2)
    d[:, 1] .= x
    d[:, 2] .= y
    N = length(x)
    v = [[(zx[j] + x[i], zy[j] + y[i]) for j in 1:6] for i in 1:N]
    h_list = [Polygon(GeometryBasics.Point2f[points for points in v[i]]) for i in 1:N]
    return h_list, v, N, d
end

# Sites grouped into concentric bands of width h around the centre
# (distance bands [delta, delta+h), ...). Empty bands are dropped.
function Hex_band(; dia=25, h=2, delta=1)
    k2 = vcat([0], ([delta:h:(dia)*2+delta;]) .^ 2)   # squared band edges
    K  = length(k2) - 1
    hh = Hexagon11(; n=dia - 1)[4]
    Nsites = size(hh, 1)
    bands = [[i for i in 1:Nsites if k2[ii] <= hh[i, 1]^2 + hh[i, 2]^2 < k2[ii+1]] for ii in 1:K]
    return bands[length.(bands) .> 0]
end

function Test_input1(; num=6, Nreleases=1, Npop=10^8, p_in=0.9, maxdist=1, rebound=true, rand_release=false, tp=Int64)
    num = num - 1
    h_list, v, Nsites, d = Hexagon1(; n=num)
    A = Hexagon_to_adj_new(; Nsites=Nsites, d=d, p_in=p_in, dia=num, maxdist=maxdist)
    if rebound
        A = [A[1], repeat([1.0], Nsites)]
    end
    for i in 1:Nsites
        A[1][i, :] = A[1][i, :] ./ sum(A[1][i, :])
    end
    Release_sites = [num + 1]
    Nmean = round(tp, Npop / Nsites)
    N = repeat([Nmean], Nsites)
    return A, Release_sites, Nsites, N, h_list, d
end

function Test_input_old(; num=6, Nreleases=1, Npop=10^8, p_in=0.9, maxdist=1, rebound=true, rand_release=false, tp=Int64)
    num = num - 1
    h_list, v, Nsites, d = Hexagon1(; n=num)
    A = Hexagon_to_adj_new(; Nsites=Nsites, d=d, p_in=p_in, dia=num, maxdist=maxdist)
    if rebound
        A = [A[1], repeat([1.0], Nsites)]
    end
    return A
end

# (A, w): dense adjacency matrix and per-site weights (all 1 when rebound=true)
function Test_input(; num=6, Nreleases=1, Npop=10^8, p_in=0.9, maxdist=1, rebound=true, rand_release=false, tp=Int64)
    num = num - 1
    h_list, v, Nsites, d = Hexagon1(; n=num)
    A = Hexagon_to_adj_new(; Nsites=Nsites, d=d, p_in=p_in, dia=num, maxdist=maxdist)
    rebound && return (Matrix{Float64}(A[1]), ones(Float64, Nsites))
    return (Matrix{Float64}(A[1]), Vector{Float64}(A[2]))
end

# Number of sites in a hexagon of radius N, and the inverse
NSites(N) = 3 * N^2 - 3 * N + 1
NSites_reverse(n) = round(Int, (sqrt(9 + 12 * (n - 1)) + 3) / 6)

function SI_inner(; Nsites, dia, d, n=0)
    dm = Matrix(d)                              # convert once, not once per site
    dv = [dm[i, :] for i in 1:Nsites]
    h  = Hexagon1(; n=n)
    hm = Matrix(h[4])
    hd = [hm[i, :] for i in 1:h[3]]
    return [[dv[j] in (hd .+ [dv[k]]) for j in 1:Nsites] for k in 1:Nsites]
end

function Hexagon_to_adj_new(; maxdist=1, Nsites, dia, d, p_in=0.99)
    n  = maxdist + 1
    si = [SI_inner(; Nsites=Nsites, dia=dia, d=d, n=nn) for nn in 0:n]
    si_diff = si[n+1] .* n .- sum(@view si[1:n])      # distance-decay exponent

    di = zeros(Nsites, Nsites)
    for i in 1:Nsites
        @inbounds for j in 1:Nsites
            if si[n][i][j] > 0
                di[i, j] = exp(-si_diff[i][j])
            end
        end
    end

    w = sum(@view di[dia+1, :])
    p_out = (1 - p_in) / (w - 1)

    rows, cols, vals = Int[], Int[], Float64[]
    for i in 1:Nsites
        total = 0.0
        for j in 1:Nsites
            if i != j && di[i, j] > 0
                val = p_out * di[i, j]
                push!(rows, i); push!(cols, j); push!(vals, val)
                total += val
            end
        end
        push!(rows, i); push!(cols, i); push!(vals, 1.0 - total)   # stay probability
    end
    A = sparse(rows, cols, vals, Nsites, Nsites)
    weights = p_in ./ diag(A)
    return A, weights
end

Site_loc(dat; dia=25, H=2) = begin
    Nr_ = Hex_band(; dia=dia, h=H)
    [[r for r in eachindex(Nr_) if (dat[j] in Nr_[r])][1] for j in eachindex(dat)]
end

# Site indices grouped by ring distance from the centre site
function Nsize_r0(; maxd=10, diam=10)
    A, Release_sites, Nsites, N, h_list, d = Test_input1(; num=diam, Nreleases=1, Npop=100,
                                                        p_in=0.95, maxdist=1, rebound=true)
    dv = collect(eachrow(d))
    center = dv[diam+1]
    hex_list = [Hexagon1(; n=dis) for dis in 1:maxd]
    hd = [collect(eachrow(hex_list[dis][4])) for dis in 1:maxd]

    sites_ind = Vector{Vector{Int}}(undef, maxd + 1)
    sites_ind[1] = [diam + 1]
    visited = BitSet(sites_ind[1])
    for dis in 1:maxd
        target_coords = [h .+ center for h in hd[dis]]
        new_sites = Int[]
        for j in 1:Nsites
            if !(j in visited) && (dv[j] in target_coords)
                push!(new_sites, j)
            end
        end
        union!(visited, new_sites)
        sites_ind[dis+1] = new_sites
    end
    return sites_ind
end


#==============================================================================
  13. UTILITIES
==============================================================================#

# Frequency of allele `a` in females, males and overall
function Marginalize_a(; Nf, Nm, a="W", M2, M, namesG)
    i1 = findall(occursin.(a, namesG))          # genotypes carrying a
    i2 = findall(occursin.(a * a, namesG))      # homozygotes (counted twice)
    pfa = sum(Nf) > 0 ? (sum(Nf[i1]) + sum(Nf[i2])) / sum(Nf) / 2 : 0.0
    pma = sum(Nm) > 0 ? (sum(Nm[i1]) + sum(Nm[i2])) / sum(Nm) / 2 : 0.0
    # FIX: old line added vectors of different lengths (DimensionMismatch)
    pa = sum(Nf .+ Nm) > 0 ? (pfa + pma) / 2 : 0.0
    return pfa, pma, pa
end

Second(x) = [x[j][2] for j in eachindex(x)]

# Merge per-replicate files into one .jld2
function merge_results(fn_prefix, rep)
    all_models = Vector{Any}(undef, rep)
    for i in 1:rep
        fn = string(fn_prefix, "_rep_", i, ".jld2")
        @load fn mod
        all_models[i] = mod
    end
    jldsave(string(fn_prefix, ".jld2"), true; models1=all_models)
    return all_models
end

function delete_files(fn_prefix, rep)
    for i in 1:rep
        fn = string(fn_prefix, "_rep_", i, ".jld2")
        try
            rm(fn; force=true)
        catch err
            @warn "Could not delete file $fn" err
        end
    end
end



# ---------- 6f. Model_sp_track: tracked allele copies in males --------------
# Output Stat1[gen, site, t], an Array{tp} of size maxgen × Nsites × (ntrack + 1):
#   t = 1..ntrack : copies of tracked allele t in males,
#                   = sum over genotypes g in Gens[t] of Gcount[t][q] * Nm[g]
#   t = ntrack+1  : total males
# Recorded at the START of each generation (before reproduction), i.e. the
# released/initial population at gen 1 and the post-migration population after.
function _track_indices(track_gnt, namesG)
    Gens   = [findall(occursin.(a, namesG)) for a in track_gnt]
    Gcount = [count.(a, namesG)[Gens[t]] for (t, a) in enumerate(track_gnt)]
    return Gens, Gcount
end

function _kernel_sp_track(model::F, Ntf_init, Ntm_init, neighbors, probs, w_scaled, wmf, E, r_sex,
                          alpha, wa, phi1, phi2, maxgen, M, M2, beta, ::Type{T},
                          Gens::Vector{Vector{Int}}, Gcount::Vector{Vector{Int}}) where {F,T}
    Nsites = size(Ntf_init, 2)
    ntr = length(Gens)
    Ntf_cur, Ntm_cur = copy(Ntf_init), copy(Ntm_init)
    Ntf_next, Ntm_next = zeros(T, M2, Nsites), zeros(T, M2, Nsites)
    sumf  = zeros(eltype(Ntf_cur), Nsites)
    Stat1 = Array{T}(undef, maxgen, Nsites, ntr + 1)
 
    for gen in 1:maxgen
        # Record male allele counts before this generation's reproduction
        @inbounds for site in 1:Nsites
            for t in 1:ntr
                g_t, c_t = Gens[t], Gcount[t]
                s = 0
                for q in eachindex(g_t)
                    s += c_t[q] * Ntm_cur[g_t[q], site]
                end
                Stat1[gen, site, t] = s
            end
            Stat1[gen, site, ntr+1] = sum(view(Ntm_cur, :, site))
        end
        # Reproduction at all sites (threaded), then migration
        _sp_reproduce!(Ntf_next, Ntm_next, sumf, Ntf_cur, Ntm_cur, model, E, alpha, wmf,
                       r_sex, phi1, phi2, wa, beta, M2, M, T)
        if gen < maxgen
            Nf_mig, Nm_mig = _migration(Ntf_next, Ntm_next, neighbors, probs, w_scaled, T)
            Ntf_cur .= Nf_mig
            Ntm_cur .= Nm_mig
        end
    end
    return Stat1
end
 
# Gens[t]: genotype indices (or a Bool mask) carrying tracked allele t;
# Gcount[t]: copies of the allele in each of those genotypes (same order).
function Model_sp_track(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
                        alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1,
                        mu_mig=1, maxdist=1, M=5, beta=0, tp=Int64, Gens, Gcount)
    M2 = div(M * (M + 1), 2)
    f  = wmf[3][1, 1]
    alpha_v = _alpha_vector(alpha, Ntf_init, Ntm_init, f, phi1, phi2)   # FIX: per-site sum
    model   = Model == 2 ? Model2 : Model1
    r_sex   = _make_r_sex(M, M2, mD_sex)
    Gens_v   = Vector{Int}[g isa AbstractVector{Bool} ? findall(g) : collect(Int, g) for g in Gens]
    Gcount_v = Vector{Int}[collect(Int, c) for c in Gcount]
    all(length.(Gens_v) .== length.(Gcount_v)) ||
        throw(ArgumentError("Gens[t] and Gcount[t] must have the same length"))
    neighbors, probs = adjacency_lists(Adj[1])
    # mu_mig scales migration as in Model_sp (original ignored it; callers use mu_mig = 1)
    return _kernel_sp_track(model, Ntf_init, Ntm_init, neighbors, probs, mu_mig .* Adj[2], wmf, E,
                            r_sex, alpha_v, wa, phi1, phi2, maxgen, M, M2, beta, tp, Gens_v,
                            Gcount_v)
end

function run_replicate_track(i; Gens, Gcount, wmf, SexDistortion, e, phi1, phi2, Model, Nmv, Nfv,
                             maxgen, Adj, Release, alpha, wa, Release_sites, NumRelease, M, homing,
                             beta, fn_prefix)
    mod = Model_sp_track(; Gens=Gens, Gcount=Gcount, Ntf_init=Nfv, Ntm_init=Nmv, Adj=Adj, wmf=wmf,
                         E=e, Model=Model, alpha=alpha, maxgen=maxgen, phi1=phi1, phi2=phi2, wa=wa,
                         mu_mig=1, Release_sites=Release_sites, maxdist=1, M=M,
                         mD_sex=SexDistortion, beta=beta)
    fn = string(fn_prefix, "_rep_", i, ".jld2")
    @save fn mod
    return fn
end

function Sp_runDistr_track(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.035, A=0,
                           Npop=10^6, rep=10, maxgen=100, dia=10, mig=0.01, Model=2,
                           Release=0.1, Release_sites=[], NumRelease=10^3, alpha=0, phi1=1, phi2=1, wa=1,
                           M=4, filename="Sp", homing=:b, beta=0, track=false, i0=1, initN=false,
                           Nfv0=[], Nmv0=[], Nfv=[], Nmv=[], track_gnt=["D", "O"])
    M2 = Int(M * (M + 1) ÷ 2)
    namesG = _names_genotypes(M == 5 ? 5 : 4)     # 15 names for M = 5, otherwise 10 (as before)
    Gens, Gcount = _track_indices(track_gnt, namesG)
 
    if isempty(Release_sites)
        Release_sites = [dia]
    end
    f = wmf[3][1, 1]
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B, c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in   = 1 .- mig
    Nsites = NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop, p_in=p_in, maxdist=1, rebound=true)
    tr  = string("_T", join("_" .* track_gnt))          # e.g. "_T_D_O"
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C, "_A_", A, "_sd_", SexDistortion, "_mig_", mig)
 
    if !initN
        Nfv, Nmv   = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites, Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C, "_A_", A, "_sd_", SexDistortion, "_mig_", mig)
    end
    if alpha == 0
        alpha = [Alpha(; phi1=phi1, phi2=phi2, f=f, N=sum(Nfv0 .+ Nmv0; dims=1)[s]) for s in 1:Nsites]
    end
    return pmap(CachingPool(workers()), i0:rep+i0-1) do i
        run_replicate_track(i; Gens=Gens, Gcount=Gcount, wmf=wmf, Nmv=Nmv, Nfv=Nfv, Model=Model,
                            SexDistortion=SexDistortion, e=e, Adj=Adj, maxgen=maxgen, alpha=alpha,
                            wa=wa, phi1=phi1, phi2=phi2, Release=Release, Release_sites=Release_sites,
                            NumRelease=NumRelease, M=M, homing=homing, beta=beta,
                            fn_prefix=fn_prefix)
    end
end