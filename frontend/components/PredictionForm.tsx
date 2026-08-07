import type { UseFormReturn } from "react-hook-form";

import type { DemoPreset } from "@/lib/presets";
import type { ScenarioFormValues } from "@/lib/form";

const countries = ["Country_A", "Country_B", "Country_C", "Country_D", "Country_E", "Country_F", "Country_G", "Country_H", "Country_I"];
const productTypes = ["Jeans", "productType_A", "productType_B", "productType_C", "productType_D", "productType_E", "productType_F", "productType_G", "productType_H", "productType_I", "productType_J"];
const brands = ["Brand_A", "Brand_B", "Brand_C", "Brand_D", "Brand_E", "Brand_F", "Brand_G", "Brand_I", "Brand_J", "Brand_K", "Pull&Bear"];

type Props = {
  form: UseFormReturn<ScenarioFormValues>;
  apiReady: boolean;
  submitting: boolean;
  selectedPreset: string;
  presets: DemoPreset[];
  onPresetChange: (id: string) => void;
  onSubmit: () => void;
};

function FieldError({ message }: { message?: string }) {
  return message ? <p className="field-error" role="alert">{message}</p> : null;
}

function BooleanSelect({
  label,
  name,
  form,
}: {
  label: string;
  name: "isMale" | "premier";
  form: UseFormReturn<ScenarioFormValues>;
}) {
  const error = form.formState.errors[name]?.message;
  return (
    <label>
      <span>{label}</span>
      <select aria-invalid={Boolean(error)} {...form.register(name)}>
        <option value="">Select</option>
        <option value="true">Yes</option>
        <option value="false">No</option>
      </select>
      <FieldError message={error} />
    </label>
  );
}

export function PredictionForm({ form, apiReady, submitting, selectedPreset, presets, onPresetChange, onSubmit }: Props) {
  const { register, watch, formState } = form;
  const customerProfileAvailable = watch("customerProfileAvailable");
  const productInformationAvailable = watch("productInformationAvailable");

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} noValidate>
      <section className="form-section" aria-labelledby="demo-heading">
        <div className="section-heading">
          <p className="eyebrow">Synthetic demo</p>
          <h2 id="demo-heading">Start with an example</h2>
        </div>
        <label>
          <span>Load example</span>
          <select value={selectedPreset} onChange={(event) => onPresetChange(event.target.value)}>
            <option value="">Choose a synthetic scenario</option>
            {presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
          </select>
        </label>
      </section>

      <fieldset className="form-section">
        <legend>
          <span className="eyebrow">Customer</span>
          Customer context
        </legend>
        <label className="toggle-row">
          <input type="checkbox" {...register("customerProfileAvailable")} />
          <span>Customer profile is available</span>
        </label>
        {customerProfileAvailable ? (
          <div className="field-grid">
            <label>
              <span>Year of birth <em>optional</em></span>
              <input inputMode="numeric" placeholder="e.g. 1988" aria-invalid={Boolean(formState.errors.yearOfBirth)} {...register("yearOfBirth")} />
              <FieldError message={formState.errors.yearOfBirth?.message} />
            </label>
            <BooleanSelect label="Male" name="isMale" form={form} />
            <label>
              <span>Shipping country</span>
              <input list="shipping-countries" placeholder="Choose or enter a value" aria-invalid={Boolean(formState.errors.shippingCountry)} {...register("shippingCountry")} />
              <datalist id="shipping-countries">{countries.map((country) => <option key={country} value={country} />)}</datalist>
              <FieldError message={formState.errors.shippingCountry?.message} />
            </label>
            <BooleanSelect label="Premier member" name="premier" form={form} />
          </div>
        ) : (
          <p className="quiet-note">A stable internal key is created only for this missing-profile scenario. It is never displayed or included in the result.</p>
        )}
      </fieldset>

      <fieldset className="form-section">
        <legend>
          <span className="eyebrow">Product</span>
          Product context
        </legend>
        <label className="toggle-row">
          <input type="checkbox" {...register("productInformationAvailable")} />
          <span>Product information is available</span>
        </label>
        {productInformationAvailable ? (
          <div className="field-grid">
            <label>
              <span>Product type <em>optional</em></span>
              <input list="product-types" placeholder="e.g. Jeans" {...register("productType")} />
              <datalist id="product-types">{productTypes.map((type) => <option key={type} value={type} />)}</datalist>
            </label>
            <label>
              <span>Brand <em>optional</em></span>
              <input list="brands" placeholder="Choose or enter a value" {...register("brandDesc")} />
              <datalist id="brands">{brands.map((brand) => <option key={brand} value={brand} />)}</datalist>
            </label>
            <label>
              <span>Average GBP price <em>optional</em></span>
              <input inputMode="decimal" placeholder="e.g. 54.99" aria-invalid={Boolean(formState.errors.avgGbpPrice)} {...register("avgGbpPrice")} />
              <FieldError message={formState.errors.avgGbpPrice?.message} />
            </label>
            <label>
              <span>Average discount <em>optional</em></span>
              <input inputMode="decimal" placeholder="0–100" aria-invalid={Boolean(formState.errors.avgDiscountValue)} {...register("avgDiscountValue")} />
              <FieldError message={formState.errors.avgDiscountValue?.message} />
            </label>
          </div>
        ) : (
          <p className="quiet-note">The frozen API accepts an unavailable product profile; the result will flag the weaker evidence.</p>
        )}
      </fieldset>

      <button className="primary-button" type="submit" disabled={!apiReady || submitting}>
        {submitting ? "Calculating score…" : apiReady ? "Calculate return-risk score" : "Waiting for API"}
      </button>
    </form>
  );
}
