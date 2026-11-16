# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# VIEW layer - turns a prediction result into spoken output
# Does: speaks the predicted item name with a confidence-aware caveat,
#       speaks several candidates if no single one is confident, speaks a
#       distinct message when nothing was detected at all
# Does not: decide what the label or confidence should be - that's the
#       model and controller's job. this file only decides how to say it
# Reasoning: the only file in the whole project the end user actually
#       experiences - a visual grid alone doesn't serve a visually-
#       impaired user, so this is its own layer, not an afterthought
#       bolted onto a cv demo. every announcement names itself as a
#       prototype, never as a verified identification.

# imports start
import pyttsx3
# imports end


# useful functions start

# creates one speech engine to be reused across an entire session - never
# call pyttsx3.init() more than once per process. repeatedly initializing
# it (once per announcement) is a known pyttsx3 reliability problem,
# especially on macOS - the second init in the same process can hang.
# callers get an engine handle back without needing to know it's pyttsx3
# specifically, keeping that detail inside the view layer.
def create_speech_engine():
    return pyttsx3.init()


# speaks the single best match, with a confidence-aware caveat rather than
# always sounding equally certain - and always names itself as a prototype,
# never as a verified identification
def announce_match(engine, class_name, confidence, low_confidence_threshold=0.6):
    if confidence < low_confidence_threshold:
        message = "I'm not confident about this one. My best guess is {}, but please double check.".format(class_name)
    else:
        message = "This looks like {}.".format(class_name)

    # every announcement carries this, regardless of confidence - the model
    # should never sound more certain than a prototype actually is
    message += " This is a computer vision prototype, not a verified identification."

    print(message)
    engine.say(message)
    engine.runAndWait()


# announces several candidates when the top result alone isn't confident
# enough to commit to a single answer
def announce_top_k(engine, results, k=3):
    names = [r['class_name'] for r in results[:k]]
    message = "I found a few possible matches: {}.".format(", ".join(names))
    message += " This is a computer vision prototype, not a verified identification."

    print(message)
    engine.say(message)
    engine.runAndWait()


# announces nothing was detected at all - a legitimate outcome (empty
# photo, background too similar to the item) that's different from a
# low-confidence match and deserves its own message rather than silence
def announce_no_items_found(engine):
    message = "I didn't detect any items in that photo. Try a plainer, darker background."
    print(message)
    engine.say(message)
    engine.runAndWait()
# useful functions end


# a tiny self-test of the wording only - no model, no photo, no controller
# involved. legitimate exception to "controllers are the only entry
# points": this checks the view in isolation, the same way you'd proofread
# a script before an actor ever performs it. one engine, reused across all
# three calls, same discipline every real caller has to follow too.
if __name__ == "__main__":
    engine = create_speech_engine()
    announce_match(engine, "a can of black beans", confidence=0.91)
    announce_match(engine, "a jar of tomato sauce", confidence=0.42)
    announce_no_items_found(engine)
