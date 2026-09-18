import type { User } from "@/client"
import { ControlField } from "@/design-system/forms/FieldMessages"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { UserMultiSelect } from "@/features/users/UserMultiSelect"

/** Who a policy applies to. Same control and wording as assigning a budget,
 *  because naming the people something applies to is the same decision.
 *
 *  Three states, not two. `null` is every caller, which is one policy with no
 *  scope. A list is the scoped case, and because a policy's key is its name
 *  plus its user, each person in it is a row of their own. An empty list is
 *  therefore scoped with nobody chosen yet, which is not "every caller" and is
 *  not something the form will submit.
 */
export function ScopePicker({
  userIds,
  users,
  onChange,
  isSettled,
}: {
  userIds: string[] | null
  users: User[]
  onChange: (userIds: string[] | null) => void
  /**
   * Whether a write under this name has already landed, which freezes the
   * scope. See the branch below for why it cannot be changed after that.
   */
  isSettled: boolean
}) {
  const isScoped = userIds !== null

  return (
    <div className="flex flex-col gap-3">
      <ControlField
        label="Applies to"
        description="A global policy resolves for every caller. A scoped one resolves only for the people named, and takes precedence over a global policy of the same name."
      />
      {isSettled ? (
        // Withheld, not disabled: there is no write that takes a policy back,
        // so a control offering to change who this applies to would be
        // offering something this form cannot do. Taking a person out of the
        // selection would leave the policy already written for them in place,
        // and choosing every caller would leave it in place AND outranking the
        // global one for exactly that person, which is the precedence rule
        // stated above. Stated rather than greyed out, because a disabled
        // control with no reason beside it teaches nothing.
        <p className="text-caption">
          Some policies under this name have already been created, and this form
          cannot take one back, so who it applies to is fixed now. Create policy
          writes the ones still missing. To remove one you did not mean to
          create, close this and delete it from the list.
        </p>
      ) : (
        <>
          <TabRow>
            {/* Each tab acts only on a change of state: pressing the one
                already active would otherwise throw away the people chosen
                under it. */}
            <Tab
              isActive={!isScoped}
              onPress={() => {
                if (isScoped) onChange(null)
              }}
            >
              Every caller
            </Tab>
            <Tab
              isActive={isScoped}
              onPress={() => {
                if (!isScoped) onChange([])
              }}
            >
              Specific users
            </Tab>
          </TabRow>
          {userIds === null ? null : (
            <UserMultiSelect
              label="Users"
              value={userIds}
              onChange={onChange}
              users={users}
              description="One policy is written per person, each resolving only for them."
            />
          )}
        </>
      )}
    </div>
  )
}
