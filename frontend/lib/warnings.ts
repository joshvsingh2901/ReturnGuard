import type { DataContext } from "@/lib/types";

const labels: Record<string, string> = {
  productType: "Product type",
  brandDesc: "Brand",
  avgGbpPrice: "Average GBP price",
  avgDiscountValue: "Average discount",
  yearOfBirth: "Birth year",
};

export function presentWarnings(dataContext: DataContext, apiWarnings: string[]): string[] {
  const warnings = [...apiWarnings];
  if (dataContext.customer_profile_imputed) {
    warnings.unshift("Customer profile was unavailable, so the model used a governed donor profile.");
  } else if (dataContext.customer_birth_year_imputed) {
    warnings.unshift("Birth year was unavailable and was imputed by the frozen preprocessing pipeline.");
  }
  if (!dataContext.product_profile_available) {
    warnings.push("Product information was unavailable; this score has weaker product evidence.");
  } else if (!dataContext.product_profile_complete) {
    const fields = dataContext.missing_product_fields.map((field) => labels[field] ?? field).join(", ");
    warnings.push(`Product information is incomplete (${fields}); this score has weaker product evidence.`);
  }
  for (const category of dataContext.unknown_categories) {
    warnings.push(`${labels[category.field] ?? category.field} is outside the observed categories and is handled safely.`);
  }
  for (const field of dataContext.outside_training_range_fields) {
    warnings.push(`${labels[field] ?? field} is outside the observed training range.`);
  }
  return [...new Set(warnings)];
}
