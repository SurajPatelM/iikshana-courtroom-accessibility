# User Manual

## Who is this system for?

Iikshana is built for courtroom participants who are blind but can hear. It provides real time spoken language assistance so that a blind participant can follow courtroom proceedings in their preferred language, hear descriptions of visual evidence, and interact with the system entirely through keyboard shortcuts and a screen reader.

This is an assistive tool. It does not replace certified court interpreters or produce official court records. Everything the system outputs should be treated as a helpful aid, not as an authoritative source.

## What the system provides

### Real time transcription and translation

The system listens to courtroom audio (through a microphone or audio feed), converts speech to text, translates it into the participant's language, and reads the translation back as spoken audio. This happens in near real time.

Each speaker in the courtroom (judge, witness, attorney) is identified separately, and the audio playback uses different voices for different speakers so the listener can tell who is talking.

### Visual evidence descriptions

When a photograph, document, or other visual exhibit is presented, the system can generate a text description of what the image contains. This description is read aloud by the screen reader or displayed in the transcript.

### Emotion indicators

The system detects vocal emotions (e.g., stressed, calm, neutral) from the original speech and can convey that emotion in the translated audio playback. This can be toggled on or off based on the participant's preference.

## Using the interface

### Getting started

1. Open the application in a web browser. The frontend runs as a Progressive Web App and works in any modern browser (Chrome, Firefox, Edge, Safari).
2. The interface is designed to work with screen readers (JAWS, NVDA, VoiceOver). It uses proper ARIA labels and live regions, so screen reader users should be able to navigate without visual reference.
3. Press **H** to open the help panel at any time for a list of available shortcuts and features.

### Keyboard shortcuts

All system functions are accessible through keyboard shortcuts. No mouse is needed.

| Key | What it does |
|-----|-------------|
| **S** | Start or stop audio capture (begin/end listening to courtroom audio) |
| **P** | Pause or resume audio playback (pause/resume the translated speech output) |
| **R** | Replay the last audio segment |
| **E** | Toggle emotion indicators on or off |
| **I** | Request a description of the currently displayed image |
| **H** | Open the help and shortcuts panel |

These shortcuts work globally, meaning they respond regardless of which part of the interface has focus.

### Accessibility controls

The control panel lets you adjust:

- **Playback speed**: slow down or speed up the translated audio output
- **Volume**: adjust the TTS playback volume independently from system volume
- **Emotion toggle**: turn emotional voice modulation on or off

All controls are labeled for screen readers and can be operated with the keyboard.

### The transcript display

The main area of the interface shows the live transcript. Each line is labeled with the speaker's role (e.g., "Judge:", "Witness:") and the translated text. New transcript entries are announced automatically through ARIA live regions, so the screen reader reads them as they appear.

The transcript also uses color coding for different speakers, which helps sighted users but does not affect the screen reader experience.

## What the system does NOT do

It is important to understand the boundaries of this system:

- It is **not** a certified assistive technology device
- It does **not** produce official court records or legal documents
- It does **not** replace the need for certified human interpreters
- Translations may contain errors, especially with uncommon legal terms or heavy accents
- Low confidence translations are flagged for review, but the system continues providing output while waiting for human verification
- The system is **always subject to human oversight**. A human interpreter can override any output at any time

## Privacy and data handling

- No live courtroom audio is stored or recorded by the system
- No audio data is sent outside the court's local network in a production deployment
- The system runs entirely on premises
- Session data (transcripts, context) exists only in memory during the active session and is discarded when the session ends
- All AI generated output is labeled as assistive, never authoritative

## Troubleshooting

**No audio playback:** Check that the browser has permission to play audio. Some browsers require a user interaction (like pressing a button) before they allow audio playback. Press S to start capture, which also initializes the audio context.

**Screen reader not announcing new transcript lines:** Make sure you are using a supported screen reader (JAWS, NVDA, or VoiceOver). The transcript uses ARIA live regions set to "polite", which means announcements wait for the screen reader to finish its current output.

**Connection lost:** If the WebSocket connection to the backend drops, the interface will show a status indicator. The system attempts to reconnect automatically. If it does not reconnect, refresh the page and press S to restart the session.

**Keyboard shortcuts not responding:** Make sure no text input field has focus. The shortcuts are global but may be intercepted if the browser focus is inside a text box. Press Escape to clear focus and try the shortcut again.
