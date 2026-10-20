"""Deterministic templates for common, well-defined churn analyses."""
import re


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _find_column(frame, question, aliases=None, numeric_only=False):
    aliases = aliases or {}
    text = " " + _normalized(question) + " "
    candidates = []
    for column in frame.columns:
        if numeric_only and not (str(frame[column].dtype).startswith(("int", "uint", "float"))
                                 or str(frame[column].dtype) in {"Int64", "Float64"}):
            continue
        names = [str(column), *aliases.get(str(column), [])]
        for name in names:
            normalized = _normalized(name)
            if normalized and f" {normalized} " in text:
                candidates.append((len(normalized), column))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def _outcome_column(frame):
    preferred = ("churned", "is_churned", "churn", "attrition", "churn_status")
    by_name = {_normalized(column): column for column in frame.columns}
    for name in preferred:
        if _normalized(name) in by_name:
            return by_name[_normalized(name)]
    return None


def deterministic_churn_code(question, frame):
    """Return a vetted code template for a supported churn question, else None.

    Templates use actual dataset column names and helpers with explicit missing
    value, denominator, and interval semantics. Unrecognized questions remain
    on the normal planner/code-generation path.
    """
    text = _normalized(question)

    if re.search(r"\b(missing|null|na values?)\b", text):
        return (
            "missing_counts = df.isna().sum()\n"
            "result = {str(column): int(count) for column, count in missing_counts.items() if count}\n"
            "print(result)"
        )

    if re.search(r"\b(duplicate|duplicates|duplicated)\b", text):
        return (
            "result = verify.duplicates(df)\n"
            "print(f\"Extra exact duplicate rows: {result['duplicate_rows']}\")\n"
            "print(f\"Rows in duplicate groups, including first copies: {result['rows_in_duplicate_groups']}\")"
        )

    target = _outcome_column(frame)
    if target is None or not re.search(r"\b(churn|churned|attrition|retention)\b", text):
        return None
    target_literal = repr(str(target))

    if re.search(r"\b(associated|association|correlat|relationship)\b", text):
        aliases = {
            "support_tickets": ["support tickets", "tickets"],
            "tenure_months": ["tenure", "months subscribed"],
            "monthly_fee": ["monthly fee", "fee"],
        }
        feature = _find_column(frame, question, aliases, numeric_only=True)
        if feature is not None and str(feature) != str(target):
            feature_literal = repr(str(feature))
            return (
                f"group_summary = stats.grouped_mean(df, value={feature_literal}, group={target_literal})\n"
                f"comparison = stats.compare_groups(df, {feature_literal}, {target_literal}, True, False)\n"
                "result = {'group_means': group_summary['groups'], 'comparison': comparison, "
                "'caveat': 'Descriptive association only; this does not establish causation.'}\n"
                "print(result)"
            )

    if re.search(r"\b(band|bands|bucket|buckets|bracket|brackets)\b", text):
        tenure_aliases = {
            "tenure_months": ["tenure", "months subscribed", "customer tenure"],
        }
        value = _find_column(frame, question, tenure_aliases, numeric_only=True)
        if value is not None and "tenure" in _normalized(value):
            value_literal = repr(str(value))
            return (
                f"result = churn.rate_by_bins(df, column={value_literal}, "
                f"bins=[-np.inf, 3, 6, 12, 24, np.inf], "
                "labels=['<3', '3-<6', '6-<12', '12-<24', '24+'], "
                f"outcome={target_literal})\n"
                "band_df = pd.DataFrame(result['bands'])\n"
                "band_df['churn_rate_percent'] = (band_df['churn_rate'] * 100).round(1)\n"
                "plot_df = band_df[band_df['n'] > 0]\n"
                "fig = px.bar(plot_df, x='band', y='churn_rate_percent', text='churn_rate_percent', "
                "title='Churn Rate by Tenure Band', "
                "labels={'band': 'Tenure Band', 'churn_rate_percent': 'Churn Rate (%)'})\n"
                "fig.update_layout(xaxis_title='Tenure Band', yaxis_title='Churn Rate (%)')\n"
                "fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')\n"
                "print(band_df.to_string(index=False, na_rep='N/A'))"
            )

    if re.search(r"\b(mean|average|avg)\b", text):
        aliases = {
            "monthly_fee": ["monthly fee", "fee"],
            "support_tickets": ["support tickets", "tickets"],
            "tenure_months": ["tenure", "months subscribed"],
        }
        value = _find_column(frame, question, aliases, numeric_only=True)
        if value is not None:
            value_literal = repr(str(value))
            return (
                f"summary = stats.grouped_mean(df, value={value_literal}, group={target_literal})\n"
                "result = summary['groups']\n"
                "for row in result:\n"
                "    label = ('Churned' if row['group'].casefold() in {'true', '1', 'yes', 'churned'} else 'Active' if row['group'].casefold() in {'false', '0', 'no', 'active', 'retained'} else 'Outcome missing')\n"
                "    mean_text = 'undefined' if row['mean'] is None else f\"{row['mean']:.2f}\"\n"
                "    print(f\"{label}: mean={mean_text}; n={row['n']}; "
                "missing excluded={row['excluded_missing']}\")"
            )

    if re.search(r"\b(median|medians)\b", text):
        aliases = {"tenure_months": ["tenure", "months subscribed"]}
        value = _find_column(frame, question, aliases, numeric_only=True)
        if value is not None:
            value_literal = repr(str(value))
            return (
                f"result = {{'active_median': df.loc[df[{target_literal}] == False, {value_literal}].median(), "
                f"'active_n': int((df[{target_literal}] == False).sum()), "
                f"'churned_median': df.loc[df[{target_literal}] == True, {value_literal}].median(), "
                f"'churned_n': int((df[{target_literal}] == True).sum())}}\n"
                "print(f\"Active: median={result['active_median']} months (n={result['active_n']})\")\n"
                "print(f\"Churned: median={result['churned_median']} months (n={result['churned_n']})\")"
            )

    segment_aliases = {
        "plan": ["plans", "contract", "subscription plan"],
        "region": ["regions", "area", "territory"],
        "payment_method": ["payment methods", "payment type"],
        "auto_renew": ["auto renewal", "auto-renewal", "automatic renewal"],
    }
    segment = _find_column(frame, question, segment_aliases)
    if segment is not None and str(segment) == str(target):
        segment = None
    if segment is not None and re.search(r"\b(compare|across|between|rank|highest|lowest|differ|show)\b", text):
        segment_literal = repr(str(segment))
        sort = re.search(r"\b(highest|lowest|rank)\b", text) is not None
        code = (
            f"summary = churn.segment_lift(df, segment={segment_literal}, outcome={target_literal})\n"
            "table = pd.DataFrame(summary['segments']).rename(columns={'segment': 'group', 'n': 'customers'})\n"
            "table['churn_rate_percent'] = (table['churn_rate'] * 100).round(1)\n"
        )
        if sort:
            code += "table = table.sort_values('churn_rate', ascending=False)\n"
        code += (
            "result = table[['group', 'customers', 'churned', 'churn_rate_percent']]\n"
            "print(result.to_string(index=False))\n"
            "print('Descriptive snapshot comparison only; this does not establish causation.')"
        )
        return code

    if re.search(r"\b(percentage|percent|count|number|balance|distribution|chart|plot)\b", text):
        code = (
            f"summary = churn.class_balance(df, outcome={target_literal})\n"
            "balance_df = pd.DataFrame(summary['classes'])\n"
            "result = summary['classes']\n"
            "for row in result:\n"
            "    print(f\"{row['class']}: {row['count']} ({row['percent']:.2f}%)\")\n"
            "print(f\"Valid denominator: {summary['n']}; outcomes excluded as missing: {summary['excluded_missing']}\")\n"
        )
        if re.search(r"\b(chart|plot|visual|visualiz)\b", text):
            code += (
                "fig = px.bar(balance_df, x='class', y='count', text='percent', color='class', "
                "title='Churn Class Balance', labels={'class': 'Churn Status', 'count': 'Customers'})\n"
                "fig.update_layout(xaxis_title='Churn Status', yaxis_title='Number of Customers', showlegend=False)\n"
                "fig.update_traces(texttemplate='%{y} (%{text:.1f}%)', textposition='outside')\n"
            )
        return code

    return None
