import { describe, expect, it } from "vitest";

import { presentWarnings } from "@/lib/warnings";

describe("presentWarnings", () => {
  it("deduplicates structured data-context warnings and uses friendly labels", () => {
    const warnings = presentWarnings(
      {
        customer_profile_imputed: true,
        customer_birth_year_imputed: false,
        product_profile_available: true,
        product_profile_complete: false,
        missing_product_fields: ["brandDesc", "avgGbpPrice"],
        unknown_categories: [{ field: "brandDesc", value: "New brand" }],
        outside_training_range_fields: ["yearOfBirth"],
      },
      ["Birth year is outside the observed training range."],
    );

    expect(warnings).toEqual([
      "Customer profile was unavailable, so the model used a governed donor profile.",
      "Birth year is outside the observed training range.",
      "Product information is incomplete (Brand, Average GBP price); this score has weaker product evidence.",
      "Brand is outside the observed categories and is handled safely.",
    ]);
  });
});
