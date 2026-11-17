# Updates & Changelog

## [1.0.2] - 2026-11-17

### Voice Chat Mode Polish

- **Increased Bottom Button Spacing**: Doubled the bottom margin of the voice recording button (16dp → 32dp) for a more balanced, comfortable layout.
- **Fixed ML Kit TTS Speaking Animation**: The model's speech-level ring previously filled to ~1/3 and stayed static during ML Kit TTS playback. Replaced the single constant amplitude with a synthetic harmonic oscillator (840ms + 480ms dual-wave) that produces organic, speech-like pulsing throughout the entire utterance.
- **Relocated Status Hint Text**: Moved the "Paused — I didn't catch that" hint text from below the caption (near the bottom button) to directly beneath the model's GemOrb, keeping status cues visually tied to the model's presence.

## [1.0.1] - 2026-11-16

### Voice Chat Mode Enhancements & Symmetrical UI Redesign

- **Muted Haptics during Voice Mode**:
  - Model response typing haptic pulses are now suppressed when Voice Chat mode is active, reserving tactile feedback and audio output purely for natural voice conversation.
- **Improved Voice Action States**:
  - The bottom record button now strictly reflects active microphone capture (`LISTENING`) instead of staying in an active state during model generation and speaking (`THINKING` / `SPEAKING`).
- **Instant Voice Turn Interruption (Barge-in)**:
  - Tapping to speak or starting voice input while the assistant is writing or speaking immediately halts active model generation and TTS playback, seamlessly transitioning into listening to the user.
- **Symmetrical Input/Output Visual Layout**:
  - **Top Model Disc (`GemOrb`)**: Re-proportioned to 104dp with balanced top margins to symmetrically mirror the user's action button below. Raw scale-pulsing was replaced with a responsive speech output energy ring that dynamically meters TTS playback amplitude.
  - **Bottom User Button**: Enlarged and repositioned to 104dp with matching bottom margins. Audio metering rings for user microphone input levels (`inputLevel`) now emanate from around this bottom button.
  - Centered conversation title and live caption text now rest naturally within the spacious layout between the model and human interface points.
