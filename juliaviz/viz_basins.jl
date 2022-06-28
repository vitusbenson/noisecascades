using DynamicalSystems 
using DifferentialEquations
using GLMakie
using IntervalArithmetic
using JLD2


# julia --project=@. -i basins.jl

# p = [c1, c2, c3, A12, A13, A21, A23, A31, A32, t1, t2, t3]

@inline @inbounds function loop(u, p, t)
    c1 = p[1]; c2 = p[2]; c3 = p[3]
    d1 = p[4]; d2 = p[5]
    d3 = p[6]; d4 = p[7]
    d5 = p[8]; d6 = p[9]
    t1 = p[10]; t2 = p[11]; t3 = p[12]
    du1 = (-u[1]^3 + u[1] + c1 + d1 * u[2] + d2 * u[3])/t1*t2
    du2 = (-u[2]^3 + u[2] + c2 + d3 * u[1] + d4 * u[3])
    du3 = (-u[3]^3 + u[3] + c3 + d5 * u[1] + d6 * u[2])/t3*t2
    return SVector{3}(du1, du2, du3)
end
# Jacobian=>
@inline @inbounds function loop_jac(u, p, t)
    c1 = p[1]; c2 = p[2]; c3 = p[3]
    d1 = p[4]; d2 = p[5]
    d3 = p[6]; d4 = p[7]
    d5 = p[8]; d6 = p[9]
    t1 = p[10]; t2 = p[11]; t3 = p[12]
    J = @SMatrix [(-3*u[1]^2+1)/t1*t2 d1/t1*t2  d2/t1*t2;
    d3 (-3*u[2]^2+1) d4;
    d5/t3*t2   d6/t3*t2  (-3*u[3]^2+1)/t3*t2]
    return J
end

params = Dict(
    "zc_0_1" => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "zc_07_1" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "zc_mix_1" => [0.26943012562182533, 0.14, 0.29, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "zc_13_1" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    "ff+_0_1" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ff-_0_1" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    "fb+_0_1" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    "fb-_0_1" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    "ff+_0_10" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 10.0, 1.0],
    "ff-_0_10" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 10.0, 1.0],
    "fb+_0_10" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 10.0, 1.0],
    "fb-_0_10" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 10.0, 1.0],
    "ff+_0_3" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 9.0],
    "ff-_0_3" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 3.0, 9.0],
    "fb+_0_3" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 9.0],
    "fb-_0_3" => [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 3.0, 19.0],
    "ff+_07_1" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ff-_07_1" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    "fb+_07_1" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    "fb-_07_1" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    "ff+_07_10" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 10.0, 1.0],
    "ff-_07_10" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 10.0, 1.0],
    "fb+_07_10" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 10.0, 1.0],
    "fb-_07_10" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 10.0, 1.0],
    "ff+_07_3" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 9.0],
    "ff-_07_3" => [0.26943012562182533, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 3.0, 9.0],
    "fb+_07_3" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 9.0],
    "fb-_07_3" => [0.26943012562182533, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 3.0, 19.0],
    "ff+_13_1" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "ff-_13_1" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1.0],
    "fb+_13_1" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    "fb-_13_1" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    "ff+_13_10" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 10.0, 1.0],
    "ff-_13_10" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 10.0, 1.0],
    "fb+_13_10" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 10.0, 1.0],
    "fb-_13_10" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 10.0, 1.0],
    "ff+_13_3" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 3.0, 9.0],
    "ff-_13_3" => [0.5003702332976757, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0, 3.0, 9.0],
    "fb+_13_3" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 3.0, 9.0],
    "fb-_13_3" => [0.5003702332976757, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 3.0, 19.0],
    "osc05" => [0.10387623521324311, 0.2698054291420028, 0.7722876027194978, -0.2336227787736659, 0.05399207923759898, 0.26525519151272064, -0.13311895627330828, 0.3350983063327801, 0.0535381553876666, 1865.4339212188884, 114.21024007462583, 913.6819205970066],
    "osc10" => [-0.07575446432282382, 0.4019416643814151, 1.1609240644399446, -0.4672455575473318, 0.10798415847519796, 0.5305103830254413, -0.26623791254661655, 0.6701966126655602, 0.1070763107753332, 1865.4339212188884, 114.21024007462583,
    913.6819205970066]
)



@load "basins.jld2" all_basins

xg = yg = zg = range(-2.0, 2.0; length = 100)
GLMakie.activate!()
with_theme(theme_dark()) do

    fig = Figure(resolution = (1080, 1080))

   
    ax = LScene(fig[1, 1]; show_axis = true)

    k = Observable{Any}("zc_0_1")

    menu = Menu(fig[2, 1], options = collect(keys(all_basins)), default = "zc_0_1")


    stablebasins = lift(k) do k
        all_basins[k]
    end


    cmapc = :VanGogh2
    cmap = Makie.categorical_colors(cmapc, 8)


    

    plt = volumeslices!(ax, xg, yg, zg, stablebasins; colormap = cmapc, colorrange = (1., 8.), transparency = true) 


    lsgrid = labelslidergrid!(
        fig,
        ["yz plane - x axis", "xz plane - y axis", "xy plane - z axis"],
        [1:length(xg), 1:length(yg), 1:length(zg)]
    )
    fig[3, 1] = lsgrid.layout

    sl_yz, sl_xz, sl_xy = lsgrid.sliders

    on(sl_yz.value) do v
        plt[:update_yz][](v)
    end
    on(sl_xz.value) do v
        plt[:update_xz][](v)
    end
    on(sl_xy.value) do v
        plt[:update_xy][](v)
    end

    set_close_to!(sl_yz, 0.0length(xg))
    set_close_to!(sl_xz, 0.0length(yg))
    set_close_to!(sl_xy, 0.0length(zg))


    on(menu.selection) do s
        k[] = s
        set_close_to!(sl_yz, sl_yz.value[])
        set_close_to!(sl_xz, sl_xz.value[])
        set_close_to!(sl_xy, sl_xy.value[])
    end
    notify(menu.selection)

    display(fig)
end