export const WAVE_PERIOD = 24
const fundamental = (2 * Math.PI) / WAVE_PERIOD

const waves = [
  { amplitude: 1, wavelength: 7.5, heading: 1.18, phase: 0 },
  { amplitude: 0.66, wavelength: 4.6, heading: 0.88, phase: 1.7 },
  { amplitude: 0.42, wavelength: 2.7, heading: 1.95, phase: 3.9 },
  { amplitude: 0.26, wavelength: 1.8, heading: 2.32, phase: 5.2 },
].map(({ amplitude, wavelength, heading, phase }) => {
  const k = (2 * Math.PI) / wavelength
  return {
    amplitude,
    phase,
    lateral: k * Math.cos(heading),
    forward: k * Math.sin(heading),
    frequency: Math.round(Math.sqrt(9.81 * k) / fundamental) * fundamental,
  }
})
const totalAmplitude = waves.reduce((total, wave) => total + wave.amplitude, 0)

/** Project continuous traveling waves onto the bar field, as on the YC reference. */
export function barLuminance(x: number, y: number, time: number, scale = 1) {
  const row = Math.min(1, Math.max(0, y))
  const depth = 1 / Math.tan(0.095 + (0.52 - 0.095) * row)
  const lateral = (x - 0.5) * 54 * 0.055 * depth
  let height = 0
  for (const wave of waves) {
    height +=
      wave.amplitude *
      Math.sin(
        (depth * wave.forward + lateral * wave.lateral) / scale -
          wave.frequency * time +
          wave.phase,
      )
  }
  const crest =
    Math.max(0, Math.min(1, 0.5 + (0.5 * height) / totalAmplitude)) ** 1.45
  const fog = 0.84 + 0.16 * (1 - row)
  return 0.06 + 0.86 * crest * fog
}
