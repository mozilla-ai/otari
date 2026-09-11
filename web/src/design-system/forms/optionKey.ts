/**
 * The prefix every select option's key carries, and the two functions that put
 * it on and take it off.
 *
 * react-aria reads an empty key as "nothing selected", and "" is a real value
 * in this product ("All", "Any price", "No workspace"), so a key of "" would
 * make an applied choice indistinguishable from an unmade one. Prefixing every
 * key sidesteps that without asking a call site to invent a sentinel.
 *
 * Shared by `forms/Select` and `navigation/FilterSelect` rather than written
 * twice: the prefix only works while both directions agree on it, and two
 * copies of a constant that has to agree is the shape of a bug that shows up as
 * a select which silently will not select.
 */
const OPTION_KEY_PREFIX = "v:"

/** A value, as the key react-aria will carry it under. */
export const optionKey = (value: string) => `${OPTION_KEY_PREFIX}${value}`

/** A key from react-aria, back as the value a call site handed over. */
export const optionValue = (key: string) => key.slice(OPTION_KEY_PREFIX.length)
