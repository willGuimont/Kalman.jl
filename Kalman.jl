import LinearAlgebra
using Plots; pyplot()

dt = 1.0
x₀ = vec([0.0; 0.0])
x̂ = x₀
P = [1.0 1.0; 1.0 1.0]
Φ = [1 dt; 0 1]
Γ = [0.5 * dt^2; dt][:, :]
Λ = [1.0 0.0][:, :]
Cᵥ = [0.1^2 0; 0 0.1^2]
Cᵨ = [35.0^2]

true_position = 0
true_speed = 0
nb_step = 200
time = 0

positions = zeros(nb_step)
speeds = zeros(nb_step)
measurements = zeros(nb_step)
estimates = zeros(nb_step)
times = zeros(nb_step)

plt = plot(
    3,
    xlim=(0, nb_step),
    ylim=(0, 150),
    title="Kalman Filter",
    label = ["True position" "Estimate" "Measures"],
    marker=2,
)

anim = @animate for i = 1:nb_step
    global true_position, true_speed, time, true_speed, x̂, P, Φ, Γ, Λ, Cᵥ, Cᵨ, positions, speeds, measurements, estimates, times

    time = time + dt
    command = 0.5 * cos(time / (nb_step * dt / 2) * π) / 10

    noised_command = command + randn() * 0.2 * LinearAlgebra.norm(command)
    true_position = true_position + true_speed * dt + 0.5 * noised_command * dt^2
    true_speed = true_speed + noised_command * dt

    u = vec([command])
    z = vec([true_position + randn() * 25])

    x̂ = Φ * x̂ + Γ * u
    P = Φ * P * Φ' + Cᵥ

    ẑ = Λ * x̂
    r = z - ẑ
    K = P * Λ' * inv(Λ * P * Λ' + Cᵨ)
    x̂ = x̂ + K * r
    P = (LinearAlgebra.I - K * Λ) * P

    push!(plt, time, [true_position, x̂[1], z[1]])

    positions[i] = true_position
    speeds[i] = true_speed
    measurements[i] = z[1]
    estimates[i] = x̂[1]
    times[i] = time
end

gif(anim, "kalman.gif", fps = 15)
savefig("kalman.pdf")
