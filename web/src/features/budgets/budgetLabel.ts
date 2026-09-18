/**
 * How a budget is named on screen, including the ones nobody named.
 *
 * A name is optional on every surface that creates a budget, and an unnamed one
 * used to render as the head of its id. That is not something a person can
 * choose by: a picker offering `04f2f38a` and `9b71c0de` offers no choice at
 * all. What an admin actually picks by is the figure and the period, so that is
 * what an unnamed budget is called here, with the id added back only where two
 * of them would still read alike.
 */

import { hasNoLimit, limitLabel, shortPeriodLabel } from "./organizationBudget"

/**
 * The fields a label is derived from, as every budget response shape carries
 * them: the deployment's `Budget` and an organization's `OrganizationBudget`
 * both assign to it structurally.
 */
export type LabelableBudget = BudgetShape & {
  budget_id: string
  name: string | null
}

/** What a label is derived from, where there is no name to use instead. */
type BudgetShape = {
  max_budget: number | null
  token_limit: number | null
  request_limit: number | null
  reset_alignment: string | null
  budget_duration_sec: number | null
}

/** The head of a budget id: the fingerprint the budgets list already shows. */
export function shortBudgetId(budgetId: string): string {
  return budgetId.split("-")[0]
}

/**
 * Whether a budget carries a name of its own, rather than needing a derived one.
 *
 * Whitespace counts as unnamed. The forms here already send `name.trim() || null`,
 * so a blank name can only arrive from another client, and rendering it would
 * put an empty string where the label belongs.
 */
export function hasBudgetName(budget: Pick<LabelableBudget, "name">): boolean {
  return (budget.name?.trim() ?? "") !== ""
}

/**
 * What a budget with no name is called: what it caps, and how often.
 *
 * Takes the caps and the period alone, so a form can offer it for a budget that
 * does not exist yet.
 */
export function unnamedBudgetLabel(budget: BudgetShape): string {
  const limit = limitLabel(budget)
  // A budget that caps nothing has no period worth saying: "No limit / month"
  // reads as a reset on a ceiling that does not exist.
  if (hasNoLimit(budget)) return limit
  const period = shortPeriodLabel(budget)
  return period === undefined ? limit : `${limit} / ${period}`
}

/**
 * How a budget is named on screen: its own name, else what distinguishes it.
 *
 * On its own, where nothing else is being named beside it. Use `budgetLabeler`
 * for a list: two unnamed budgets capping the same figure over the same period
 * read alike here, and that is what a picker must not do.
 */
export function budgetLabel(budget: LabelableBudget): string {
  const name = budget.name?.trim()
  return name ? name : unnamedBudgetLabel(budget)
}

/**
 * A labeler for one list of budgets, which is what a picker or a table wants.
 *
 * Two unnamed budgets capping the same figure over the same period read alike,
 * and picking the wrong one silently hands somebody the wrong ceiling, so each
 * of them is qualified by the head of its id. Which ones collide is a fact about
 * the whole list, so the list is walked once here rather than rescanned per
 * label: `budgets` is every budget the deployment holds, and a picker over a
 * thousand of them cannot afford to be quadratic. A budget the list does not
 * carry still gets its plain label.
 */
export function budgetLabeler(
  budgets: readonly LabelableBudget[],
): (budget: LabelableBudget) => string {
  const derivedCounts = budgets.reduce((counts, budget) => {
    if (hasBudgetName(budget)) return counts
    const derived = unnamedBudgetLabel(budget)
    return counts.set(derived, (counts.get(derived) ?? 0) + 1)
  }, new Map<string, number>())
  return (budget) => {
    if (hasBudgetName(budget)) return budgetLabel(budget)
    const derived = unnamedBudgetLabel(budget)
    return (derivedCounts.get(derived) ?? 0) > 1
      ? `${derived} (${shortBudgetId(budget.budget_id)})`
      : derived
  }
}
